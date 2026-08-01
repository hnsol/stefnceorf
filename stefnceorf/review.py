"""レビュー用 A/B 試聴画面 (sc review)。

編集後 .sc.txt から `build_edit_plan()` を1回だけ実行し、同じ EditPlan から
「レビュー音声（実レンダー）」と「元時刻→編集後時刻の写像表」の両方を作る。
生成物は隣接する2ファイル:

- `{base}.review.wav`: 現在の .sc.txt を実レンダーした音声（`.edited.wav` とは別名）
- `{base}.review.html`: 原音とレビュー音声の2つの <audio> を相対パスで参照する単一 HTML

読み取り専用の判断支援に徹する（.sc.txt への書き戻しは持たない）。写像は実レンダーと
同じサンプル丸め（render.plan_chunk_bounds）とクロスフェード縮みを再現するため、
HTML 上のクリック位置は実際のレビュー音声と一致する。
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from dataclasses import dataclass
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import quote

from .render import (
    FADE_S,
    FILLER_CLOSE,
    FILLER_OPEN,
    GAP_MAX_S,
    GAP_THRESHOLD_S,
    BLOCK_SEP,
    EditPlan,
    _format_mmss,
    build_edit_plan,
    format_stats,
    parse_edited_txt,
    plan_chunk_bounds,
    render_plan,
)

# 出力ファイル名のサフィックス（既存出力 .edited.wav / .logic.fcpxml と衝突させない）
REVIEW_WAV_SUFFIX = ".review.wav"
REVIEW_HTML_SUFFIX = ".review.html"

# 行/単語クリック時に再生する前後の余白（秒）。カット境界の前後を耳で追える長さ。
PLAY_PAD_S = 2.0

# Logic 用 FCPXML との差異を明示する注記（HTML 冒頭に常時表示）
LOGIC_NOTE = (
    "この A/B は sc render の結果に対して正確です。"
    "Logic 用 FCPXML は削除をタイムラインの空白として残すため、"
    "Logic 上の再生とは異なります。"
)


@dataclass(frozen=True)
class MapEntry:
    """原音区間と編集後出力区間の対応（すべてサンプル単位）。"""

    segment_id: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int


def chunk_output_starts(
    bounds: list[tuple[int, int]],
    flags: list[bool],
    fade_samples: int,
) -> tuple[list[int], int]:
    """各チャンクの出力開始サンプルと総出力サンプル数を返す（純関数）。

    crossfade_concat の長さ計算をそのまま再現する。クロスフェード境界では
    f = min(fade_samples, これまでの累積長, 次チャンク長) だけ重なるため、
    次チャンクの開始は「累積長 - f」になり総長は f だけ縮む。単純結合
    （flags[k-1] が False）の境界は重ならない。

    flags の長さは len(bounds) - 1。範囲外はクロスフェード扱い（crossfade_concat と同じ）。
    """
    starts: list[int] = []
    total = 0
    for k, (s0, s1) in enumerate(bounds):
        n = s1 - s0
        if k == 0:
            start = 0
        else:
            do_cf = flags[k - 1] if k - 1 < len(flags) else True
            fade = min(fade_samples, total, n) if do_cf else 0
            start = total - fade
        starts.append(start)
        total = start + n
    return starts, total


def segment_source_spans(segments: list[dict]) -> list[tuple[str, float, float]]:
    """json の各セグメントの原音占有区間 (id, start, end) を json 順で返す（純関数）。

    区間の取り方は build_edit_plan の `ordered` と同一にする
    （未認識行は source_start/source_end、通常行は先頭 word の start 〜 末尾 word の end）。
    words が空の通常行は音を持たないため除外する。
    """
    spans: list[tuple[str, float, float]] = []
    for seg in segments:
        words = seg.get("words", []) or []
        if seg.get("kind") == "unrecognized":
            start = float(seg["source_start"])
            end = float(seg["source_end"])
        elif words:
            start = float(words[0]["start"])
            end = float(words[-1]["end"])
        else:
            continue
        spans.append((seg["id"], start, end))
    return spans


def build_mapping(
    spans: list[tuple[str, float, float]],
    bounds: list[tuple[int, int]],
    starts: list[int],
    samplerate: int,
) -> list[MapEntry]:
    """セグメント原音区間 × 出力チャンクの共通部分から写像表を作る（純関数）。

    出力チャンク順（＝編集後の時間順）に、その内側で json 順にセグメントを走査する。
    1セグメントがカットで分断された場合は複数エントリになる。
    """
    entries: list[MapEntry] = []
    for k, (s0, s1) in enumerate(bounds):
        for seg_id, start, end in spans:
            a = max(s0, int(round(start * samplerate)))
            b = min(s1, int(round(end * samplerate)))
            if b <= a:
                continue
            entries.append(
                MapEntry(
                    segment_id=seg_id,
                    source_start=a,
                    source_end=b,
                    output_start=starts[k] + (a - s0),
                    output_end=starts[k] + (b - s0),
                )
            )
    return entries


def map_source_sample(
    bounds: list[tuple[int, int]], starts: list[int], sample: int
) -> int | None:
    """原音サンプル位置を編集後サンプル位置へ写す（純関数）。

    残っているチャンクの内側なら対応位置をそのまま返す。カットされて残っていない
    位置（削除フィラー等）は最も近いチャンクの手前側端／奥側端、すなわちその削除の
    結合境界を返す。チャンクが1つも無ければ None。
    """
    if not bounds:
        return None
    best: int | None = None
    best_distance: int | None = None
    for k, (s0, s1) in enumerate(bounds):
        if s0 <= sample < s1:
            return starts[k] + (sample - s0)
        if sample < s0:
            candidate = starts[k]
            distance = s0 - sample
        else:
            candidate = starts[k] + (s1 - s0)
            distance = sample - s1 + 1
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = candidate
    return best


def _deleted_boundary(
    spans: list[tuple[str, float, float]],
    first_output: dict[str, int],
    last_output: dict[str, int],
    index: int,
) -> int:
    """削除されたセグメントの「結合境界」（編集後サンプル位置）を返す（純関数）。

    json 順で後ろ側の最初の生存セグメントの出力開始（＝削除後に続けて鳴る音の頭）を
    採る。後ろに生存が無ければ前側の最後の生存セグメントの出力終端、どちらも
    無ければ 0（全削除）。
    """
    for seg_id, _s, _e in spans[index + 1:]:
        if seg_id in first_output:
            return first_output[seg_id]
    for seg_id, _s, _e in reversed(spans[:index]):
        if seg_id in last_output:
            return last_output[seg_id]
    return 0


def _unrecognized_label(seg: dict, duration: float) -> str:
    """未認識/未転写発話セグメントの表示ラベル（transcribe の既定 txt 行と同文言）。

    UTS（reason:"uts"）は既定でカット指定の 〔〕 付き表示にする
    （--no-auto-uts 時の旧文言は json に区別情報が無いため反映しない）。
    """
    if seg.get("reason") == "uts":
        return f"{FILLER_OPEN}未転写発話 {duration:.1f}秒{FILLER_CLOSE}"
    return f"⚠ 未認識区間 {duration:.1f}秒（音声保持）"


def build_review_data(
    plan: EditPlan,
    data: dict,
    kept_ids: set[str],
    source_audio: str,
    review_audio: str,
    serve: bool = False,
    requested: dict | None = None,
    token: str | None = None,
) -> dict:
    """EditPlan と json から HTML へ渡すレビューデータを構築する。

    サンプル単位で写像を組み、HTML へは秒（float）で渡す。単語の残存判定は写像表の
    被覆で行う（diff ロジックを複製せず、実レンダー結果と必ず整合させるため）。

    削除の状態は2値に分ける（unsafe 判定で「TXT は削除指定だが EditPlan は保持」と
    いう正当な矛盾が起きるため）:

    - requested_delete: TXT 上で削除指定されているか（行なし／〔〕あり）
    - effective_delete: EditPlan で実際に音が消えたか（安全判定通過後）

    requested は serve モードで raw TXT から算出したものを
    {"segments": {id: bool}, "words": {id: {word_index: bool}}} で渡す。
    None のときは TXT 上の行の有無・写像の被覆から推定する（静的モード）。
    """
    samplerate = plan.samplerate
    segments = data.get("segments", [])
    bounds, flags, kept_interval_indices = plan_chunk_bounds(plan)
    fade_samples = int(round(FADE_S * samplerate))
    starts, total_out = chunk_output_starts(bounds, flags, fade_samples)

    spans = segment_source_spans(segments)
    entries = build_mapping(spans, bounds, starts, samplerate)

    first_output: dict[str, int] = {}
    last_output: dict[str, int] = {}
    for entry in entries:
        first_output.setdefault(entry.segment_id, entry.output_start)
        last_output[entry.segment_id] = entry.output_end

    span_index = {seg_id: i for i, (seg_id, _s, _e) in enumerate(spans)}
    span_by_id = {seg_id: (s, e) for seg_id, s, e in spans}

    def _sec(sample: int | None) -> float:
        return (sample / samplerate) if (sample is not None and samplerate) else 0.0

    req_segments: dict[str, bool] = (requested or {}).get("segments", {})
    req_words: dict[str, dict] = (requested or {}).get("words", {})

    out_segments: list[dict] = []
    for index, seg in enumerate(segments):
        seg_id = seg["id"]
        source_start, source_end = span_by_id.get(seg_id, (0.0, 0.0))
        deleted = seg_id not in kept_ids
        has_audio = seg_id in first_output
        if has_audio:
            output = first_output[seg_id]
        elif seg_id in span_index:
            output = _deleted_boundary(
                spans, first_output, last_output, span_index[seg_id]
            )
        else:
            output = 0

        # 未認識/UTS 行の精密カットは前後にポーズを残すため、島の一部（マージン分）が
        # 出力に残る。「音が1サンプルでも残っているか」では削除を検出できないので、
        # 島の中央が残っているかで判定する（単語の cut 判定と同じ考え方）。
        if seg.get("kind") == "unrecognized":
            seg_mid = int(round((source_start + source_end) / 2.0 * samplerate))
            effective_delete = not any(s0 <= seg_mid < s1 for s0, s1 in bounds)
        else:
            effective_delete = not has_audio

        seg_req_words = req_words.get(seg_id, {})
        words_out: list[dict] = []
        for word_index, word in enumerate(seg.get("words", []) or []):
            w_start = word.get("start")
            w_end = word.get("end")
            if w_start is None or w_end is None:
                continue
            w_start = float(w_start)
            w_end = float(w_end)
            mid = int(round((w_start + w_end) / 2.0 * samplerate))
            covered = any(s0 <= mid < s1 for s0, s1 in bounds)
            head = int(round(w_start * samplerate))
            word_req = seg_req_words.get(
                word_index, seg_req_words.get(str(word_index), not covered)
            )
            words_out.append(
                {
                    "index": word_index,
                    "text": word.get("word", "").replace(BLOCK_SEP, ""),
                    "source": w_start,
                    "center": (w_start + w_end) / 2.0,
                    "output": _sec(map_source_sample(bounds, starts, head)),
                    "suggest": bool(word.get("suggest")),
                    "cut": not covered,
                    "requested_delete": bool(word_req),
                    "effective_delete": not covered,
                }
            )

        out_segments.append(
            {
                "id": seg_id,
                "index": index,
                "kind": seg.get("kind", "speech"),
                "reason": seg.get("reason"),
                "deleted": deleted,
                "requested_delete": bool(req_segments.get(seg_id, deleted)),
                "effective_delete": effective_delete,
                "toggle": (
                    "bracket"
                    if seg.get("kind") == "unrecognized" and seg.get("reason") == "uts"
                    else "line"
                ),
                "ts": _format_mmss(source_start),
                "source": source_start,
                "source_end": source_end,
                "output": _sec(output),
                "words": words_out,
                "label": (
                    _unrecognized_label(seg, max(0.0, source_end - source_start))
                    if seg.get("kind") == "unrecognized"
                    else seg.get("text", "")
                ),
            }
        )

    # 無音トリム箇所。plan.silence_trim_boundaries は output_intervals の index 空間なので
    # 空区間を落としたチャンク index 空間へ写し、隣り合うチャンクの境界だけを採る。
    interval_to_chunk = {v: k for k, v in enumerate(kept_interval_indices)}
    trims: list[dict] = []
    for boundary in sorted(plan.silence_trim_boundaries):
        k = interval_to_chunk.get(boundary)
        nxt = interval_to_chunk.get(boundary + 1)
        if k is None or nxt is None or nxt != k + 1:
            continue
        gap_start = bounds[k][1]
        gap_end = bounds[nxt][0]
        trims.append(
            {
                "index": len(trims),
                "source": _sec(gap_start),
                "source_end": _sec(gap_end),
                "output": _sec(starts[nxt]),
                "removed": _sec(gap_end - gap_start),
                "ts": _format_mmss(_sec(gap_start)),
            }
        )

    return {
        "base": plan.base_name,
        "serve": bool(serve),
        "token": token,
        "source_audio": source_audio,
        "review_audio": review_audio,
        "pad": PLAY_PAD_S,
        "samplerate": samplerate,
        "source_samples": plan.total_samples,
        "output_samples": total_out,
        "output_duration": _sec(total_out),
        "segments": out_segments,
        "trims": trims,
        "stats": format_stats(plan.stats),
        "warnings": list(plan.warnings),
        "note": LOGIC_NOTE,
    }


def mapping_total_samples(plan: EditPlan) -> int:
    """写像表から算出した総出力サンプル数（実レンダー出力長との照合用）。"""
    bounds, flags, _ = plan_chunk_bounds(plan)
    _starts, total = chunk_output_starts(
        bounds, flags, int(round(FADE_S * plan.samplerate))
    )
    return total


def _embed_json(obj: dict) -> str:
    """JSON を <script> へ埋め込める形に無害化する（純関数）。

    `</` は `<\\/` へ（JSON の `\\/` は仕様上の正当なエスケープで値は不変）、
    HTML コメント開始 `<!--` は `<` を \\u003c へ置換する。どちらも JSON.parse で
    元の文字列に戻る。
    """
    text = json.dumps(obj, ensure_ascii=False)
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def _relative_media(target: Path, html_dir: Path) -> str:
    """HTML から音声を参照するための相対 URL を返す（純関数）。"""
    rel = os.path.relpath(str(target), start=str(html_dir))
    return quote(rel.replace(os.sep, "/"))


def state_label(requested: bool, effective: bool) -> str:
    """requested / effective の組み合わせから状態表示文言を返す（純関数）。

    ON かつ EditPlan でも削除 → 「削除」、ON だが安全判定で残った → 「unsafeのため保持」。
    OFF なのに音が無い（手でテキストを消した等）→ 「削除（手動編集）」。
    """
    if requested and effective:
        return "削除"
    if requested and not effective:
        return "unsafeのため保持"
    if effective:
        return "削除（手動編集）"
    return "保持"


def _segment_markup(seg: dict, serve: bool = False) -> str:
    """1セグメント分のトランスクリプト行 HTML（テキストは HTML エスケープ）。

    取消線は effective_delete（EditPlan で実際に消えたか）で付ける。serve モードでは
    requested_delete を操作するチェックボックスと状態表示を足す。
    """
    classes = ["line"]
    if seg["effective_delete"]:
        classes.append("deleted")
    # 削除指定したが安全判定で保持された状態は専用色で示す（チェックだけでは
    # 「自分で外した保持」と区別できないため）
    if seg["requested_delete"] and not seg["effective_delete"]:
        classes.append("unsafe")
    if seg["kind"] == "unrecognized":
        classes.append("unrec")
    if seg["reason"] == "uts":
        classes.append("uts")

    parts = [
        f'<div class="{" ".join(classes)}" data-i="{seg["index"]}">',
    ]
    if serve:
        checked = " checked" if seg["requested_delete"] else ""
        parts.append(
            f'<input type="checkbox" class="del" data-i="{seg["index"]}"'
            f' aria-label="この行を削除"{checked}>'
        )
    parts.extend([
        f'<span class="ts">{escape(seg["ts"])}</span>',
        f'<span class="sid">[{escape(seg["id"])}]</span>',
        '<span class="text">',
    ])
    if seg["words"]:
        for wi, word in enumerate(seg["words"]):
            wclasses = ["w"]
            if word["suggest"]:
                wclasses.append("suggest")
            if word["cut"]:
                wclasses.append("cut")
            if word.get("requested_delete") and not word["cut"]:
                wclasses.append("unsafe")
            span = (
                f'<span class="{" ".join(wclasses)}" data-w="{wi}">'
                f'{escape(word["text"])}</span>'
            )
            if serve and word["suggest"]:
                wchecked = " checked" if word["requested_delete"] else ""
                span = (
                    f'<label class="wt"><input type="checkbox" class="delw"'
                    f' data-i="{seg["index"]}" data-w="{wi}"'
                    f' aria-label="このフィラーを削除"{wchecked}>{span}</label>'
                )
            parts.append(span)
    else:
        parts.append(escape(seg["label"]))
    parts.append("</span>")
    if serve:
        parts.append(
            f'<span class="state" data-i="{seg["index"]}">'
            f'{escape(state_label(seg["requested_delete"], seg["effective_delete"]))}'
            "</span>"
        )
    parts.append("</div>")
    return "".join(parts)


def _trim_markup(trim: dict) -> str:
    """無音トリム箇所のマーカー HTML。"""
    return (
        f'<div class="trim" data-t="{trim["index"]}">'
        f'<span class="ts">{escape(trim["ts"])}</span>'
        f'<span class="sid">[--]</span>'
        f'<span class="text">✂ 無音トリム -{trim["removed"]:.1f}秒</span>'
        "</div>"
    )


def _transcript_markup(review: dict) -> str:
    """トランスクリプト全体（無音トリムマーカーを原音時刻順に差し込む）。"""
    trims = sorted(review["trims"], key=lambda t: t["source"])
    serve = bool(review.get("serve"))
    blocks: list[str] = []
    ti = 0
    for seg in review["segments"]:
        while ti < len(trims) and trims[ti]["source"] < seg["source"]:
            blocks.append(_trim_markup(trims[ti]))
            ti += 1
        blocks.append(_segment_markup(seg, serve))
    for trim in trims[ti:]:
        blocks.append(_trim_markup(trim))
    return "\n".join(blocks)


_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, "Hiragino Sans", sans-serif;
  margin: 0 auto; padding: 1rem; max-width: 60rem; line-height: 1.7;
}
h1 { font-size: 1.2rem; margin: 0 0 .1rem; }
.subtitle { font-size: .85rem; opacity: .7; margin: 0 0 .5rem; }
.note {
  font-size: .85rem; padding: .5rem .75rem; margin: 0 0 1rem;
  border-left: 3px solid #c90; background: rgba(204,153,0,.12);
}
/* 操作エリアはスクロールしても画面上部に固定する（Canvas は color-scheme 追従の背景色） */
.controls {
  position: sticky; top: 0; z-index: 10; background: Canvas;
  padding: .5rem 0 .25rem; border-bottom: 1px solid rgba(127,127,127,.4);
  margin-bottom: .75rem;
}
.players { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: .75rem; }
.players div { display: flex; align-items: center; gap: .5rem; }
.players audio { height: 2rem; }
.modes { margin-bottom: .75rem; }
.modes button {
  font: inherit; padding: .25rem .75rem; margin-right: .5rem; cursor: pointer;
  border: 1px solid currentColor; background: transparent; border-radius: .25rem;
}
.modes button.active { background: #0a64c8; color: #fff; border-color: #0a64c8; }
.legend { font-size: .8rem; margin: 0 0 1rem; padding: 0; list-style: none; }
.legend li { display: inline-block; margin-right: 1rem; }
.stats {
  font-size: .8rem; margin: 0 0 1rem; padding: .5rem .75rem;
  background: rgba(127,127,127,.12); white-space: pre-wrap;
}
#transcript { border-top: 1px solid rgba(127,127,127,.4); }
.line, .trim {
  display: flex; gap: .5rem; align-items: baseline; cursor: pointer;
  padding: .15rem .25rem; border-bottom: 1px solid rgba(127,127,127,.15);
}
.line:hover, .trim:hover { background: rgba(10,100,200,.1); }
.line.active, .trim.active { background: rgba(10,100,200,.22); }
.ts { font-variant-numeric: tabular-nums; opacity: .6; font-size: .8rem; min-width: 3.5rem; }
.sid { opacity: .4; font-size: .75rem; min-width: 3.5rem; }
.text { flex: 1; }
/* 削除済み行（.sc.txt に無い行）は取消線 */
.line.deleted .text { text-decoration: line-through; opacity: .55; }
.line.unrec .text { color: #b06000; }
.line.uts { background: rgba(255,160,0,.14); }
.w { padding: 0 .05em; border-radius: .15em; }
.w:hover { outline: 1px solid rgba(10,100,200,.6); }
/* フィラー提案単語（suggest）とカットされた単語 */
.w.suggest { background: rgba(200,0,140,.16); }
.w.cut { text-decoration: line-through; opacity: .5; }
/* 削除指定したが安全判定で保持された状態（.line.uts より後に置き優先させる） */
.w.unsafe, .line.unsafe { background: rgba(220,60,40,.18); }
.trim { color: #0a7d50; font-size: .85rem; }
/* --serve（Hlyst 対話モード）専用 */
.save { display: flex; align-items: center; gap: .75rem; margin-bottom: .75rem; }
.save button {
  font: inherit; padding: .25rem 1rem; cursor: pointer; border-radius: .25rem;
  border: 1px solid currentColor; background: transparent;
}
.save button:disabled { opacity: .4; cursor: default; }
.save button.dirty { background: #0a7d50; color: #fff; border-color: #0a7d50; }
.save .msg { font-size: .8rem; }
.save .msg.error { color: #c02020; }
.state { font-size: .75rem; opacity: .65; min-width: 8rem; text-align: right; }
label.wt { display: inline; }
label.wt input { margin: 0 .1em 0 .2em; vertical-align: middle; }
.line > input.del { margin: 0; align-self: center; }
</style>
</head>
<body>
<h1>$title</h1>
<p class="subtitle">$subtitle</p>
<p class="note">$note</p>
<div class="controls">
<div class="players">
  <div><span>原音</span><audio id="audio-src" controls preload="metadata" src="$source_audio"></audio></div>
  <div><span>編集後</span><audio id="audio-out" controls preload="metadata" src="$review_audio"></audio></div>
</div>
<div class="modes">
  <button type="button" class="mode active" data-mode="src">原音を再生</button>
  <button type="button" class="mode" data-mode="out">編集後を再生</button>
</div>
$save_area
<ul class="legend">
  <li><span class="w suggest">フィラー提案</span></li>
  <li><span class="w cut">カット済み</span></li>
  <li><span class="line uts" style="display:inline">未転写発話候補</span></li>
  <li><span class="w unsafe">削除指定→安全判定で保持</span></li>
  <li><span class="trim" style="display:inline">✂ 無音トリム</span></li>
  <li>行・単語をクリックすると前後 $pad 秒を再生します</li>
</ul>
</div>
<pre class="stats">$stats</pre>
<div id="transcript">
$transcript
</div>
<script id="sc-data" type="application/json">$data</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("sc-data").textContent);
  var players = {
    src: document.getElementById("audio-src"),
    out: document.getElementById("audio-out")
  };
  var mode = "src";
  var stopAt = null;
  var current = null;

  function play(t) {
    var audio = players[mode];
    var other = players[mode === "src" ? "out" : "src"];
    other.pause();
    var from = Math.max(0, t - DATA.pad);
    stopAt = t + DATA.pad;
    current = audio;
    try { audio.currentTime = from; } catch (e) { /* metadata 未読込 */ }
    audio.play();
  }

  Object.keys(players).forEach(function (key) {
    players[key].addEventListener("timeupdate", function () {
      if (current === this && stopAt !== null && this.currentTime >= stopAt) {
        this.pause();
        stopAt = null;
      }
    });
  });

  function setActive(el) {
    var prev = document.querySelector(".active[data-i], .active[data-t]");
    if (prev) { prev.classList.remove("active"); }
    el.classList.add("active");
  }

  document.getElementById("transcript").addEventListener("click", function (ev) {
    // --serve のチェックボックス操作は change ハンドラ（試聴）に任せる
    if (ev.target.closest("input, label.wt")) { return; }
    var trim = ev.target.closest(".trim");
    if (trim) {
      setActive(trim);
      var td = DATA.trims[Number(trim.dataset.t)];
      play(mode === "src" ? td.source : td.output);
      return;
    }
    var line = ev.target.closest(".line");
    if (!line) { return; }
    setActive(line);
    var seg = DATA.segments[Number(line.dataset.i)];
    var word = ev.target.closest(".w");
    var target = word ? seg.words[Number(word.dataset.w)] : seg;
    play(mode === "src" ? target.source : target.output);
  });

  Array.prototype.forEach.call(
    document.querySelectorAll(".modes button"),
    function (btn) {
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(
          document.querySelectorAll(".modes button"),
          function (b) { b.classList.remove("active"); }
        );
        btn.classList.add("active");
        mode = btn.dataset.mode;
      });
    }
  );
})();
</script>
$serve_script
</body>
</html>
""")


# --serve 時だけ差し込むコントロール（保存ボタン＋試聴プレイヤー＋通知欄）
_SAVE_AREA = """<div class="save">
  <button type="button" id="save" disabled>保存</button>
  <audio id="audio-preview" controls preload="none"></audio>
  <span class="msg" id="save-msg"></span>
</div>"""


# --serve 時だけ差し込む対話スクリプト（トグル→即試聴→保存）
_SERVE_SCRIPT = """<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("sc-data").textContent);
  if (!DATA.serve) { return; }

  var preview = document.getElementById("audio-preview");
  var saveBtn = document.getElementById("save");
  var msg = document.getElementById("save-msg");
  var dirty = false;
  var busy = false;
  var baseline = snapshot();

  function snapshot() {
    var s = {};
    DATA.segments.forEach(function (seg) {
      s["L" + seg.id] = seg.requested_delete;
      seg.words.forEach(function (w) {
        if (w.suggest) { s["W" + seg.id + ":" + w.index] = w.requested_delete; }
      });
    });
    return s;
  }

  function toggles() {
    var out = [];
    DATA.segments.forEach(function (seg) {
      out.push({ type: "line", id: seg.id, delete: seg.requested_delete });
      seg.words.forEach(function (w) {
        if (w.suggest) {
          out.push({
            type: "word", id: seg.id, word: w.index, delete: w.requested_delete
          });
        }
      });
    });
    return out;
  }

  function setDirty() {
    var cur = snapshot();
    dirty = Object.keys(cur).some(function (k) { return cur[k] !== baseline[k]; });
    saveBtn.disabled = !dirty || busy;
    saveBtn.classList.toggle("dirty", dirty && !busy);
  }

  function note(text, isError) {
    msg.textContent = text;
    msg.classList.toggle("error", !!isError);
  }

  function stateLabel(requested, effective) {
    if (requested && effective) { return "削除"; }
    if (requested) { return "unsafeのため保持"; }
    if (effective) { return "削除（手動編集）"; }
    return "保持";
  }

  function applyStates() {
    DATA.segments.forEach(function (seg) {
      var line = document.querySelector('.line[data-i="' + seg.index + '"]');
      if (!line) { return; }
      line.classList.toggle("deleted", seg.effective_delete);
      line.classList.toggle("unsafe", seg.requested_delete && !seg.effective_delete);
      var box = line.querySelector("input.del");
      if (box) { box.checked = seg.requested_delete; }
      var state = line.querySelector(".state");
      if (state) {
        state.textContent = stateLabel(seg.requested_delete, seg.effective_delete);
      }
      seg.words.forEach(function (w) {
        var span = line.querySelector('.w[data-w="' + w.index + '"]');
        if (span) {
          span.classList.toggle("cut", w.effective_delete);
          span.classList.toggle("unsafe", w.requested_delete && !w.effective_delete);
        }
        var wbox = line.querySelector('input.delw[data-w="' + w.index + '"]');
        if (wbox) { wbox.checked = w.requested_delete; }
      });
    });
  }

  function post(path, body) {
    return fetch(path + "?token=" + encodeURIComponent(DATA.token), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Hlyst-Token": DATA.token
      },
      body: JSON.stringify(body)
    }).then(function (res) {
      return res.json().then(function (payload) {
        if (!res.ok || payload.error) {
          throw new Error(payload.error || ("HTTP " + res.status));
        }
        return payload;
      });
    });
  }

  function findWord(seg, index) {
    var found = null;
    seg.words.forEach(function (w) { if (w.index === index) { found = w; } });
    return found;
  }

  function mergeStates(segments) {
    var byId = {};
    segments.forEach(function (s) { byId[s.id] = s; });
    DATA.segments.forEach(function (seg) {
      var fresh = byId[seg.id];
      if (!fresh) { return; }
      seg.requested_delete = fresh.requested_delete;
      seg.effective_delete = fresh.effective_delete;
      seg.output = fresh.output;
      seg.words.forEach(function (w) {
        var fw = findWord(fresh, w.index);
        if (!fw) { return; }
        w.requested_delete = fw.requested_delete;
        w.effective_delete = fw.effective_delete;
        w.output = fw.output;
      });
    });
  }

  function toBlob(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i += 1) { bytes[i] = bin.charCodeAt(i); }
    return new Blob([bytes], { type: "audio/wav" });
  }

  function runPreview(target) {
    if (busy) { return; }
    busy = true;
    saveBtn.disabled = true;
    note("試聴を生成中…", false);
    post("/preview", { toggles: toggles(), target: target })
      .then(function (payload) {
        mergeStates(payload.segments);
        applyStates();
        note(payload.message || "試聴", false);
        if (preview.dataset.blob) { URL.revokeObjectURL(preview.dataset.blob); }
        preview.dataset.blob = URL.createObjectURL(toBlob(payload.audio));
        preview.src = preview.dataset.blob;
        var playing = preview.play();
        if (playing && playing.catch) { playing.catch(function () {}); }
      })
      .catch(function (err) { note(err.message, true); })
      .then(function () { busy = false; setDirty(); });
  }

  document.getElementById("transcript").addEventListener("change", function (ev) {
    var box = ev.target;
    if (!box.classList || !box.dataset) { return; }
    var seg = DATA.segments[Number(box.dataset.i)];
    if (!seg) { return; }
    if (box.classList.contains("del")) {
      seg.requested_delete = box.checked;
      setDirty();
      runPreview({ type: "line", id: seg.id });
    } else if (box.classList.contains("delw")) {
      var word = findWord(seg, Number(box.dataset.w));
      if (!word) { return; }
      word.requested_delete = box.checked;
      setDirty();
      runPreview({ type: "word", id: seg.id, word: word.index });
    }
  });

  saveBtn.addEventListener("click", function () {
    if (busy || !dirty) { return; }
    busy = true;
    saveBtn.disabled = true;
    note("保存中…", false);
    post("/save", { toggles: toggles() })
      .then(function (payload) {
        mergeStates(payload.segments);
        applyStates();
        document.querySelector(".stats").textContent = payload.stats.join("\\n");
        var out = document.getElementById("audio-out");
        out.src = DATA.review_audio + "&v=" + payload.version;
        out.load();
        baseline = snapshot();
        note("保存しました: " + payload.txt_path, false);
      })
      .catch(function (err) { note(err.message, true); })
      .then(function () { busy = false; setDirty(); });
  });

  applyStates();
  setDirty();
})();
</script>"""


def render_html(review: dict) -> str:
    """レビューデータから単一 HTML を生成する（純関数、外部依存なし）。

    serve=True のときだけ保存コントロールと対話スクリプトを差し込む
    （静的モードの出力は従来どおり読み取り専用）。
    """
    serve = bool(review.get("serve"))
    return _HTML_TEMPLATE.substitute(
        title=escape(f"Hlyst — {review['base']}"),
        subtitle=escape(
            "A/B試聴・編集（チェックで削除ON/OFF→即試聴→保存）"
            if serve else "A/B試聴"
        ),
        save_area=_SAVE_AREA if serve else "",
        serve_script=_SERVE_SCRIPT if serve else "",
        note=escape(review["note"]),
        source_audio=escape(review["source_audio"], quote=True),
        review_audio=escape(review["review_audio"], quote=True),
        pad=f"{review['pad']:g}",
        stats=escape("\n".join(review["stats"])),
        transcript=_transcript_markup(review),
        data=_embed_json(review),
    )


def review(
    txt_path: str,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
    open_browser: bool = True,
    stats_out: list[str] | None = None,
    serve: bool = False,
    port: int = 0,
) -> dict:
    """レビュー音声と A/B 試聴 HTML を生成し、生成物のパスを返す。

    戻り値 {"wav_path": ..., "html_path": ...}。stats_out が渡された場合は
    format_stats の行をそこへ append するだけで print しない（render と同じ流儀）。

    serve=True のときは HTML ファイルを書かず、127.0.0.1 のローカルサーバー
    （Hlyst 対話モード）を起こして Ctrl-C まで待ち受ける
    （戻り値は {"wav_path": ..., "url": ...}）。
    """
    if serve:
        from .serve import serve_review

        return serve_review(
            txt_path,
            gap_threshold=gap_threshold,
            gap_max=gap_max,
            open_browser=open_browser,
            port=port,
        )

    plan = build_edit_plan(txt_path, gap_threshold=gap_threshold, gap_max=gap_max)

    out_dir = plan.txt_path.parent
    wav_path = out_dir / f"{plan.base_name}{REVIEW_WAV_SUFFIX}"
    html_path = out_dir / f"{plan.base_name}{REVIEW_HTML_SUFFIX}"

    render_plan(plan, wav_path)

    json_path = out_dir / f"{plan.base_name}.sc.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    kept_ids = {
        seg_id
        for seg_id, _text, _brackets in parse_edited_txt(
            plan.txt_path.read_text(encoding="utf-8")
        )
    }

    review_data = build_review_data(
        plan,
        data,
        kept_ids,
        source_audio=_relative_media(plan.source_wav, out_dir),
        review_audio=_relative_media(wav_path, out_dir),
    )
    html_path.write_text(render_html(review_data), encoding="utf-8")

    for warning in plan.warnings:
        print(f"警告: {warning}", file=sys.stderr)

    if stats_out is not None:
        stats_out.extend(format_stats(plan.stats))
    else:
        for line in format_stats(plan.stats):
            print(line)

    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())

    return {"wav_path": str(wav_path), "html_path": str(html_path)}
