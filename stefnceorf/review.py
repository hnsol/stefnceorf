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


def state_label(
    requested: bool, effective: bool, kind: str = "speech", reason: str | None = None
) -> str:
    """requested / effective の組み合わせから状態表示文言を返す（純関数）。

    ON かつ EditPlan でも削除 → 「削除」。ただし reason が "uts" でない
    unrecognized 行（無印の未認識区間）は安全判定を通らない無条件カットなので
    「削除（無条件）」と区別する。ON だが安全判定で残った → 「unsafeのため保持」。
    OFF なのに音が無い（手でテキストを消した等）→ 「削除（手動編集）」。
    """
    if requested and effective:
        if kind == "unrecognized" and reason != "uts":
            return "削除（無条件）"
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
        is_unsafe = seg["requested_delete"] and not seg["effective_delete"]
        state_classes = "state unsafe" if is_unsafe else "state"
        label = state_label(
            seg["requested_delete"], seg["effective_delete"], seg["kind"], seg["reason"]
        )
        parts.append(
            f'<span class="{state_classes}" data-i="{seg["index"]}">'
            f'{escape(label)}'
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
/* Flexoki カラースキーム（light/dark）をカスタムプロパティで切替 */
:root {
  color-scheme: light dark;
  --bg: #FFFCF0;           /* paper */
  --surface: #F2F0E5;      /* base-50 */
  --border: #DAD8CE;       /* base-150 */
  --text: #100F0F;         /* black */
  --text-muted: #6F6E69;   /* base-600 */
  --accent: #205EA6;       /* blue-600 */
  --accent-tint-weak: rgba(32,94,166,.10);
  --accent-tint-strong: rgba(32,94,166,.22);
  --on-accent: #FFFCF0;
  --magenta: #A02F6F;      /* magenta-600 */
  --magenta-tint: rgba(160,47,111,.14);
  --orange: #BC5215;       /* orange-600 */
  --orange-tint: rgba(188,82,21,.14);
  --green: #66800B;        /* green-600 */
  --red: #AF3029;          /* red-600 */
  --yellow: #AD8301;       /* yellow-600 */
  --yellow-tint: rgba(173,131,1,.14);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #100F0F;
    --surface: #1C1B1A;    /* base-950 */
    --border: #403E3C;     /* base-800 */
    --text: #CECDC3;       /* base-200 */
    --text-muted: #B7B5AC; /* base-300 */
    --accent: #4385BE;     /* blue-400 */
    --accent-tint-weak: rgba(67,133,190,.16);
    --accent-tint-strong: rgba(67,133,190,.30);
    --magenta: #CE5D97;    /* magenta-400 */
    --magenta-tint: rgba(206,93,151,.20);
    --orange: #DA702C;     /* orange-400 */
    --orange-tint: rgba(218,112,44,.20);
    --green: #879A39;      /* green-400 */
    --red: #D14D41;        /* red-400 */
  }
}
body {
  font-family: -apple-system, "Hiragino Sans", sans-serif;
  margin: 0 auto; padding: 1rem; max-width: 60rem; line-height: 1.7;
  background: var(--bg); color: var(--text);
}
h1 { font-size: 1.2rem; margin: 0 0 .1rem; }
.subtitle { font-size: .85rem; color: var(--text-muted); margin: 0 0 .5rem; }
.note {
  font-size: .85rem; padding: .5rem .75rem; margin: 0 0 1rem;
  border-left: 3px solid var(--yellow); background: var(--yellow-tint);
}
/* 操作エリアはスクロールしても画面上部に固定する */
.controls {
  position: sticky; top: 0; z-index: 10; background: var(--surface);
  padding: .5rem 0 .25rem; border-bottom: 1px solid var(--border);
  margin-bottom: .75rem;
}
.player-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: .5rem;
  margin-bottom: .5rem;
}
.player-row audio { height: 2rem; }
.seek { display: flex; gap: .25rem; }
.mode {
  font: inherit; padding: .25rem .75rem; cursor: pointer;
  border: 1px solid currentColor; background: transparent; border-radius: .25rem;
}
.mode.active { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
.seek-btn {
  font: inherit; font-size: .75rem; padding: .15rem .5rem; cursor: pointer;
  border: 1px solid var(--border); background: transparent; border-radius: .25rem;
  color: var(--text);
}
.seek-btn:hover { border-color: var(--accent); }
.legend { font-size: .8rem; margin: 0 0 1rem; padding: 0; list-style: none; }
.legend li { display: inline-block; margin-right: 1rem; }
.stats {
  font-size: .8rem; margin: 0 0 1rem; padding: .5rem .75rem;
  background: var(--surface); border: 1px solid var(--border); white-space: pre-wrap;
}
#transcript { border-top: 1px solid var(--border); }
.line, .trim {
  display: flex; gap: .5rem; align-items: baseline; cursor: pointer;
  padding: .15rem .25rem; border-bottom: 1px solid var(--border);
}
.line:hover, .trim:hover { background: var(--accent-tint-weak); }
.line.active, .trim.active { background: var(--accent-tint-strong); }
.ts { font-variant-numeric: tabular-nums; color: var(--text-muted); font-size: .8rem; min-width: 3.5rem; }
.sid { color: var(--text-muted); font-size: .75rem; min-width: 3.5rem; opacity: .7; }
.text { flex: 1; }
/* 削除済み行（.sc.txt に無い行）は取消線 */
.line.deleted .text { text-decoration: line-through; opacity: .55; }
.line.unrec .text { color: var(--orange); }
.line.uts { background: var(--orange-tint); }
.w { padding: 0 .05em; border-radius: .15em; }
.w:hover { outline: 1px solid var(--accent); }
/* フィラー提案単語（suggest）とカットされた単語 */
.w.suggest { background: var(--magenta-tint); }
.w.cut { text-decoration: line-through; opacity: .5; }
/* 削除指定したが安全判定で保持された状態は破線アウトライン＋状態ラベル赤太字で示す
   （フィラー提案の塗りつぶしと表現チャネルを分け、.line.uts より後に置き優先させる） */
.w.unsafe, .line.unsafe {
  outline: 1px dashed var(--red); outline-offset: 1px; background: none;
}
.trim { color: var(--green); font-size: .85rem; }
/* --serve（Hlyst 対話モード）専用 */
/* 保存ボタンは1段目右端に、余白/区切りで原音プレイヤーと分離する */
.save-area {
  display: flex; align-items: center; gap: .5rem; margin-left: auto;
  padding-left: .75rem; border-left: 1px solid var(--border);
}
#save {
  font: inherit; padding: .25rem 1rem; cursor: pointer; border-radius: .25rem;
  border: 1px solid currentColor; background: transparent;
}
#save:disabled { opacity: .5; cursor: default; }
#save.dirty { background: var(--green); color: var(--on-accent); border-color: var(--green); }
.msg { font-size: .8rem; }
.msg.error { color: var(--red); }
/* 3段目: 試聴範囲切替・絶対時刻・現在案プレビュー */
.preview-panel {
  display: flex; flex-wrap: wrap; align-items: center; gap: .4rem 1rem;
  margin-bottom: .75rem;
}
.pad-select { display: flex; gap: .25rem; }
.padbtn {
  font: inherit; font-size: .75rem; padding: .15rem .6rem; cursor: pointer;
  border: 1px solid var(--border); background: transparent; border-radius: .25rem;
  color: var(--text);
}
.padbtn.active { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
.abs-times {
  display: flex; flex-wrap: wrap; gap: .2rem 1rem; font-size: .8rem;
  font-variant-numeric: tabular-nums;
}
.abs-times .abs-row { white-space: nowrap; color: var(--text-muted); }
.abs-times .abs-row b { color: var(--text); font-weight: 400; }
.state { font-size: .75rem; color: var(--text-muted); min-width: 8rem; text-align: right; }
.state.unsafe { color: var(--red); font-weight: 700; }
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
<div class="player-row">
  <button type="button" class="mode" data-mode="src">原音を再生</button>
  <audio id="audio-src" controls preload="metadata" src="$source_audio"></audio>
  <span class="seek">
    <button type="button" class="seek-btn" data-player="src" data-seek="-10">−10s</button>
    <button type="button" class="seek-btn" data-player="src" data-seek="-5">−5s</button>
    <button type="button" class="seek-btn" data-player="src" data-seek="5">＋5s</button>
    <button type="button" class="seek-btn" data-player="src" data-seek="10">＋10s</button>
  </span>
$save_area
</div>
<div class="player-row">
  <button type="button" class="mode active" data-mode="out">保存済み編集後を再生</button>
  <audio id="audio-out" controls preload="metadata" src="$review_audio"></audio>
  <span class="seek">
    <button type="button" class="seek-btn" data-player="out" data-seek="-10">−10s</button>
    <button type="button" class="seek-btn" data-player="out" data-seek="-5">−5s</button>
    <button type="button" class="seek-btn" data-player="out" data-seek="5">＋5s</button>
    <button type="button" class="seek-btn" data-player="out" data-seek="10">＋10s</button>
  </span>
</div>
$preview_panel
<ul class="legend">
  <li><span class="w suggest">フィラー提案</span></li>
  <li><span class="w cut">カット済み</span></li>
  <li><span class="line uts" style="display:inline">未転写発話候補</span></li>
  <li><span class="state unsafe" style="display:inline">削除指定→安全判定で保持</span></li>
  <li><span class="trim" style="display:inline">✂ 無音トリム</span></li>
  <li>行・単語をクリックすると前後 $pad 秒を再生します</li>
$extra_legend
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
  var mode = "out";
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
    document.querySelectorAll(".mode"),
    function (btn) {
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(
          document.querySelectorAll(".mode"),
          function (b) { b.classList.remove("active"); }
        );
        btn.classList.add("active");
        mode = btn.dataset.mode;
      });
    }
  );

  // シークボタン（原音・編集後それぞれの currentTime を加減、0〜duration にクランプ）
  Array.prototype.forEach.call(
    document.querySelectorAll(".seek-btn"),
    function (btn) {
      btn.addEventListener("click", function () {
        var audio = players[btn.dataset.player];
        if (!audio) { return; }
        var dur = isFinite(audio.duration) ? audio.duration : Infinity;
        var next = audio.currentTime + Number(btn.dataset.seek);
        audio.currentTime = Math.min(Math.max(0, next), dur);
      });
    }
  );
})();
</script>
$serve_script
</body>
</html>
""")


# --serve 時だけ差し込む保存ボタン（1段目右端、原音プレイヤーと余白/区切りで分離）
_SAVE_AREA = """<span class="save-area">
  <button type="button" id="save" disabled>保存済み</button>
  <span class="msg" id="save-msg"></span>
</span>"""


# --serve 時だけ差し込む3段目（試聴範囲切替・絶対時刻・現在案プレビュー）
_PREVIEW_PANEL = """<div class="preview-panel">
  <span class="pad-select">
    <button type="button" class="padbtn" data-pad="2">±2秒</button>
    <button type="button" class="padbtn active" data-pad="4">±4秒</button>
    <button type="button" class="padbtn" data-pad="8">±8秒</button>
  </span>
  <span class="abs-times">
    <span class="abs-row">対象 原音 <b id="abs-target">--:--.-</b></span>
    <span class="abs-row">現在案 <b id="abs-center">--:--.-</b></span>
    <span class="abs-row">試聴範囲 <b id="abs-window">--:--.- – --:--.-（±4秒）</b></span>
    <span class="abs-row">再生位置 <b id="abs-pos">--:--.-</b></span>
  </span>
  <audio id="audio-preview" controls preload="none"></audio>
  <span class="msg" id="preview-msg"></span>
</div>"""


# --serve 時だけ差し込む凡例（クリック操作の割り当てをショートカット無しでも分かるように）
_EXTRA_LEGEND = (
    "<li>クリック=現在案を試聴／Shift+クリック=保存済み編集後／"
    "Option(Alt)+クリック=原音</li>"
)


# --serve 時だけ差し込む対話スクリプト（トグル→即試聴→保存）
_SERVE_SCRIPT = """<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("sc-data").textContent);
  if (!DATA.serve) { return; }

  var preview = document.getElementById("audio-preview");
  var audioSrc = document.getElementById("audio-src");
  var audioOut = document.getElementById("audio-out");
  var saveBtn = document.getElementById("save");
  var saveMsg = document.getElementById("save-msg");
  var previewMsg = document.getElementById("preview-msg");
  var absTarget = document.getElementById("abs-target");
  var absCenter = document.getElementById("abs-center");
  var absWindow = document.getElementById("abs-window");
  var absPos = document.getElementById("abs-pos");
  var dirty = false;
  var busy = false;
  var baseline = snapshot();
  // 3段目「現在案プレビュー」の試聴範囲（秒）。3段目付近のボタンで切替できる。
  var currentPad = 4;
  // 直前にプレビューした対象（行/単語）。試聴範囲を切り替えたときの再生成に使う。
  var currentTarget = null;
  // audio-preview の再生位置表示（window_start_s + currentTime）の基準値。
  var currentWindowStart = 0;
  // トグル直後は自動再生せず、取得した音声をキャッシュして
  // 対象の行/単語をクリックしたときに再生する（key -> {url, windowStart, ...}）。
  // キーには pad を含める（対象・pad・編集状態が一致する場合のみ再利用する）。
  var previewCache = {};
  // /preview, /save の応答が世代遅れ（その後に新しい操作が始まっている）なら
  // 状態を巻き戻さないよう、リクエストごとに世代IDを振って照合する。
  var generation = 0;

  function previewKey(target, pad) {
    var base = target.type === "word"
      ? "W" + target.id + ":" + target.word
      : "L" + target.id;
    return base + "@" + pad;
  }

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

  function dirtyKeys() {
    var cur = snapshot();
    return Object.keys(cur).filter(function (k) { return cur[k] !== baseline[k]; });
  }

  // 保存ボタンの表示を busy／変更あり（件数入り）／変更なしの3状態で切り替える。
  function updateSaveButton() {
    if (busy) {
      saveBtn.disabled = true;
      saveBtn.classList.remove("dirty");
      saveBtn.textContent = "保存中…";
      return;
    }
    var n = dirtyKeys().length;
    dirty = n > 0;
    saveBtn.disabled = !dirty;
    saveBtn.classList.toggle("dirty", dirty);
    saveBtn.textContent = dirty ? "変更を保存（" + n + "件）" : "保存済み";
  }

  // busy 中は全チェックボックスと保存ボタンを操作不能にする（見た目にも分かる
  // ようにする）。busy を素通しして黙って古い操作を捨てる経路を作らないため。
  function setControlsBusy(value) {
    busy = value;
    Array.prototype.forEach.call(
      document.querySelectorAll("input.del, input.delw"),
      function (box) { box.disabled = value; }
    );
    updateSaveButton();
  }

  // previewCache の全 object URL を revoke して破棄する。トグル変更のたびに
  // 呼ぶ（他項目のトグルで編集後プランが変わり、古いキャッシュ音声は現在の
  // 計画と対応しなくなるため。単純な全破棄方式で陳腐化を防ぐ）。
  function clearPreviewCache() {
    Object.keys(previewCache).forEach(function (key) {
      URL.revokeObjectURL(previewCache[key].url);
      delete previewCache[key];
    });
  }

  function notePreview(text, isError) {
    previewMsg.textContent = text;
    previewMsg.classList.toggle("error", !!isError);
  }

  function noteSave(text, isError) {
    saveMsg.textContent = text;
    saveMsg.classList.toggle("error", !!isError);
  }

  // 秒 → "M:SS.d" 形式（等幅数字表示欄で使う）。
  function formatTime(t) {
    if (typeof t !== "number" || !isFinite(t)) { return "--:--.-"; }
    var sign = t < 0 ? "-" : "";
    t = Math.abs(t);
    var m = Math.floor(t / 60);
    var s = t - m * 60;
    var sStr = s.toFixed(1);
    if (s < 10) { sStr = "0" + sStr; }
    return sign + m + ":" + sStr;
  }

  // キャッシュ済み/応答直後の試聴データから絶対時刻表示を更新する。
  function updateAbsTimes(entry) {
    currentWindowStart = entry.windowStart;
    absTarget.textContent = formatTime(entry.target);
    absCenter.textContent = formatTime(entry.center);
    absWindow.textContent =
      formatTime(entry.windowStart) + "–" + formatTime(entry.windowEnd) +
      "（±" + Math.round(entry.pad) + "秒）";
    absPos.textContent = formatTime(entry.windowStart);
  }

  function stateLabel(requested, effective, kind, reason) {
    if (requested && effective) {
      if (kind === "unrecognized" && reason !== "uts") { return "削除（無条件）"; }
      return "削除";
    }
    if (requested) { return "unsafeのため保持"; }
    if (effective) { return "削除（手動編集）"; }
    return "保持";
  }

  function applyStates() {
    DATA.segments.forEach(function (seg) {
      var line = document.querySelector('.line[data-i="' + seg.index + '"]');
      if (!line) { return; }
      var isUnsafe = seg.requested_delete && !seg.effective_delete;
      line.classList.toggle("deleted", seg.effective_delete);
      line.classList.toggle("unsafe", isUnsafe);
      var box = line.querySelector("input.del");
      if (box) { box.checked = seg.requested_delete; }
      var state = line.querySelector(".state");
      if (state) {
        state.textContent = stateLabel(
          seg.requested_delete, seg.effective_delete, seg.kind, seg.reason
        );
        state.classList.toggle("unsafe", isUnsafe);
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

  // トグル反映済みの現在案から試聴音声を取得する（現在選択中の対象として記録し、
  // 応答が来たら audio-preview へ設定して再生を試みる）。
  function runPreview(target) {
    if (busy) { return; }
    currentTarget = target;
    var myGeneration = ++generation;
    setControlsBusy(true);
    notePreview("試聴を生成中…", false);
    post("/preview", { toggles: toggles(), target: target, pad_s: currentPad })
      .then(function (payload) {
        // この応答より後に新しい操作（トグル/保存）が始まっていたら、
        // 画面の状態を巻き戻さないよう破棄する。
        if (myGeneration !== generation) { return; }
        mergeStates(payload.segments);
        applyStates();
        notePreview((payload.message || "試聴") + "（クリックで再生）", false);
        // 対象の行/単語をクリックしたときにも再生できるよう音声をキャッシュ
        // しておく（従来どおり）。加えて、チェック変更の結果をすぐ耳で確認
        // できるよう audio-preview へも即座に設定し load/play を試みる
        // （autoplay 拒否は握りつぶし、src 設定と load だけは完了させる）。
        var key = previewKey(target, currentPad);
        var url = URL.createObjectURL(toBlob(payload.audio));
        var entry = {
          url: url,
          windowStart: payload.window_start_s,
          windowEnd: payload.window_end_s,
          center: payload.center_s,
          target: payload.target_source_s,
          pad: payload.pad_s
        };
        previewCache[key] = entry;
        updateAbsTimes(entry);
        preview.src = url;
        preview.load();
        var playing = preview.play();
        if (playing && playing.catch) { playing.catch(function () {}); }
      })
      .catch(function (err) {
        if (myGeneration !== generation) { return; }
        notePreview(err.message, true);
      })
      .then(function () {
        if (myGeneration !== generation) { return; }
        setControlsBusy(false);
      });
  }

  // 現在案プレビューを再生する（対象＋現在の試聴範囲でキャッシュ済みなら即再生、
  // 無ければ現在のチェック状態で /preview を生成してから再生する）。
  function playCurrentPlan(target) {
    currentTarget = target;
    var key = previewKey(target, currentPad);
    var cached = previewCache[key];
    if (cached) {
      updateAbsTimes(cached);
      preview.src = cached.url;
      var playing = preview.play();
      if (playing && playing.catch) { playing.catch(function () {}); }
      return;
    }
    runPreview(target);
  }

  function seekAndPlay(audio, t) {
    try { audio.currentTime = t; } catch (e) { /* metadata 未読込 */ }
    var playing = audio.play();
    if (playing && playing.catch) { playing.catch(function () {}); }
  }

  function setActiveLine(el) {
    var prev = document.querySelector(".line.active, .trim.active");
    if (prev) { prev.classList.remove("active"); }
    el.classList.add("active");
  }

  // 行/単語クリックの再生先を切り替える（capture フェーズで先取りし、本文
  // スクリプト側の既定動作＝上段2プレイヤーへは渡さない）。
  // 通常クリック＝3段目「現在案プレビュー」、Shift+クリック＝保存済み編集後、
  // Option(Alt)+クリック＝原音。無音トリムは対象が無いため上段プレイヤーの
  // 既定動作（本文スクリプト側）に委ねる。
  document.getElementById("transcript").addEventListener("click", function (ev) {
    if (ev.target.closest("input, label.wt")) { return; }
    if (ev.target.closest(".trim")) { return; }
    var line = ev.target.closest(".line");
    if (!line) { return; }
    var seg = DATA.segments[Number(line.dataset.i)];
    if (!seg) { return; }
    var word = ev.target.closest(".w");
    var wordData = word ? findWord(seg, Number(word.dataset.w)) : null;

    ev.stopPropagation();
    setActiveLine(line);

    if (ev.shiftKey) {
      seekAndPlay(audioOut, wordData ? wordData.output : seg.output);
      return;
    }
    if (ev.altKey) {
      seekAndPlay(audioSrc, wordData ? wordData.source : seg.source);
      return;
    }

    var target = wordData
      ? { type: "word", id: seg.id, word: wordData.index }
      : { type: "line", id: seg.id };
    playCurrentPlan(target);
  }, true);

  // 試聴範囲（±2/±4/±8秒）の切替。現在選択中の対象があれば再生成する。
  Array.prototype.forEach.call(
    document.querySelectorAll(".padbtn"),
    function (btn) {
      btn.addEventListener("click", function () {
        var pad = Number(btn.dataset.pad);
        if (pad === currentPad) { return; }
        currentPad = pad;
        Array.prototype.forEach.call(
          document.querySelectorAll(".padbtn"),
          function (b) { b.classList.remove("active"); }
        );
        btn.classList.add("active");
        if (currentTarget) { playCurrentPlan(currentTarget); }
      });
    }
  );

  // 再生位置表示（3段目）。window_start_s + currentTime を M:SS.d で更新する。
  preview.addEventListener("timeupdate", function () {
    absPos.textContent = formatTime(currentWindowStart + preview.currentTime);
  });

  document.getElementById("transcript").addEventListener("change", function (ev) {
    var box = ev.target;
    if (!box.classList || !box.dataset) { return; }
    var seg = DATA.segments[Number(box.dataset.i)];
    if (!seg) { return; }
    if (box.classList.contains("del")) {
      seg.requested_delete = box.checked;
      updateSaveButton();
      clearPreviewCache();
      runPreview({ type: "line", id: seg.id });
    } else if (box.classList.contains("delw")) {
      var word = findWord(seg, Number(box.dataset.w));
      if (!word) { return; }
      word.requested_delete = box.checked;
      updateSaveButton();
      clearPreviewCache();
      runPreview({ type: "word", id: seg.id, word: word.index });
    }
  });

  saveBtn.addEventListener("click", function () {
    if (busy || !dirty) { return; }
    var myGeneration = ++generation;
    setControlsBusy(true);
    post("/save", { toggles: toggles() })
      .then(function (payload) {
        if (myGeneration !== generation) { return; }
        mergeStates(payload.segments);
        applyStates();
        document.querySelector(".stats").textContent = payload.stats.join("\\n");
        var out = document.getElementById("audio-out");
        out.src = DATA.review_audio + "&v=" + payload.version;
        out.load();
        baseline = snapshot();
        // 保存成功後は編集後プランが確定するため、古いプレビュー音声を
        // すべて破棄する（保存前の中間状態の音が再生されないようにする）。
        clearPreviewCache();
        noteSave("保存しました: " + payload.txt_path, false);
      })
      .catch(function (err) {
        if (myGeneration !== generation) { return; }
        noteSave(err.message, true);
      })
      .then(function () {
        if (myGeneration !== generation) { return; }
        setControlsBusy(false);
      });
  });

  applyStates();
  updateSaveButton();
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
            "A/B試聴・編集（チェックで削除ON/OFF→クリックで試聴→保存）"
            if serve else "A/B試聴"
        ),
        save_area=_SAVE_AREA if serve else "",
        preview_panel=_PREVIEW_PANEL if serve else "",
        extra_legend=_EXTRA_LEGEND if serve else "",
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
