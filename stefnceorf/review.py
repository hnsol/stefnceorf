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

# Hlyst の h1 直下の見出し・サブタイトル（serve/静的モード共通）
HLYST_H1 = "Hlyst — A/B試聴・編集"
HLYST_SUBTITLE = "削除のON/OFFを切り替え、現在案を試聴して保存します。"

# Logic 用 FCPXML との差異を明示する注記（export ボタンの title 属性にだけ表示）
EXPORT_NOTE = (
    "Hlystは sc render の音声を再生します。"
    "Logic用FCPXMLではカット部分が空白として残るため、間の長さは一致しません。"
    "最終確認はLogicで行ってください。"
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

    未認識/UTS（reason:"uts"）はどちらも既定でカット指定の 〔〕 付き表示にする
    （--no-auto-uts 時の旧文言は json に区別情報が無いため反映しない）。
    """
    if seg.get("reason") == "uts":
        return f"{FILLER_OPEN}未転写発話 {duration:.1f}秒{FILLER_CLOSE}"
    return f"{FILLER_OPEN}未認識 {duration:.1f}秒{FILLER_CLOSE}"


def compute_display_order(kept_order: list[str], data: dict) -> list[str]:
    """画面の表示順（全セグメント ID の順列）を返す（純関数）。

    TXT に残っている行はその並び（＝出力順）をそのまま使い、削除済みの行は
    serve.apply_toggles の復活位置（_insert_position）と同じ規則で差し込む:
    json 順で直後にくる生存行の前、無ければ直前の生存行の後ろ、どちらも無ければ
    末尾。削除済みが連続する場合は json 順に並ぶ（先に挿入した行も基準になる）。
    """
    json_order = [seg["id"] for seg in data.get("segments", [])]
    kept = [seg_id for seg_id in kept_order if seg_id in set(json_order)]
    out = list(kept)
    placed = set(out)
    for i, seg_id in enumerate(json_order):
        if seg_id in placed:
            continue
        pos = len(out)
        for other in json_order[i + 1:]:
            if other in placed:
                pos = out.index(other)
                break
        else:
            for other in reversed(json_order[:i]):
                if other in placed:
                    pos = out.index(other) + 1
                    break
        out.insert(pos, seg_id)
        placed.add(seg_id)
    return out


def build_review_data(
    plan: EditPlan,
    data: dict,
    kept_ids: set[str],
    source_audio: str,
    review_audio: str,
    serve: bool = False,
    requested: dict | None = None,
    token: str | None = None,
    display_order: list[str] | None = None,
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

    display_order は画面の表示順（compute_display_order の戻り値）。segments は
    json 順のまま（data-i の意味を変えない）で、表示順は payload の "order" が
    持つ。None のときは json 順をそのまま表示順にする。
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

        # 比較窓のシーク計算専用の中心位置。保持対象は map_source_sample() で
        # 編集後位置へ写像し、削除対象は既存の output（結合境界）をそのまま使う
        # （リストの表示用タイムスタンプは source/output の開始時刻のまま変えない）。
        source_center = (source_start + source_end) / 2.0
        if effective_delete:
            output_center = _sec(output)
        else:
            center_sample = int(round(source_center * samplerate))
            output_center = _sec(map_source_sample(bounds, starts, center_sample))

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
                    "output_center": _sec(map_source_sample(bounds, starts, mid)),
                    "suggest": bool(word.get("suggest")),
                    "cut": not covered,
                    "requested_delete": bool(word_req),
                    "effective_delete": not covered,
                }
            )

        requested_delete_val = bool(req_segments.get(seg_id, deleted))
        unsafe_reason = plan.guarded_cut_reasons.get(seg_id)
        out_segments.append(
            {
                "id": seg_id,
                "index": index,
                "kind": seg.get("kind", "speech"),
                "reason": seg.get("reason"),
                "deleted": deleted,
                "requested_delete": requested_delete_val,
                "effective_delete": effective_delete,
                "unsafe_reason": unsafe_reason,
                "state": state_label(
                    requested_delete_val,
                    effective_delete,
                    seg.get("kind", "speech"),
                    seg.get("reason"),
                    unsafe_reason,
                ),
                "toggle": (
                    "bracket" if seg.get("kind") == "unrecognized" else "line"
                ),
                "ts": _format_mmss(source_start),
                "source": source_start,
                "source_end": source_end,
                "source_center": source_center,
                "output": _sec(output),
                "output_center": output_center,
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

    # 表示順は必ず全 ID の順列にする（欠けている ID は json 順で末尾へ足す）。
    json_ids = [seg["id"] for seg in segments]
    order_ids: list[str] = []
    if display_order:
        known = set(json_ids)
        taken: set[str] = set()
        for seg_id in display_order:
            if seg_id in known and seg_id not in taken:
                order_ids.append(seg_id)
                taken.add(seg_id)
        order_ids.extend(seg_id for seg_id in json_ids if seg_id not in taken)
    else:
        order_ids = list(json_ids)

    return {
        "base": plan.base_name,
        "serve": bool(serve),
        "order": order_ids,
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
    requested: bool,
    effective: bool,
    kind: str = "speech",
    reason: str | None = None,
    unsafe_reason: str | None = None,
) -> str:
    """requested / effective の組み合わせから状態表示文言を返す（純関数）。

    未認識/UTS の削除も guarded cut（元位置ポーズ0.25秒の確保が条件）になった
    ため、ON かつ EditPlan でも削除 → 常に「削除」（無条件カット専用の文言は
    廃止）。ON だが安全判定で残った場合は unsafe_reason で文言を分ける:
    "pause"（元位置のポーズを確保できない）→「ポーズ不足のため保持」、それ
    以外（境界RMS等）→「unsafeのため保持」。OFF なのに音が無い（手でテキスト
    を消した等）→「削除（手動編集）」。kind/reason は将来の分岐用に引数として
    残すが、現状の判定には使わない。保持（無指定・音あり）は例外ではないため
    表示しない（空文字）。
    """
    if requested and effective:
        return "削除"
    if requested and not effective:
        if unsafe_reason == "pause":
            return "ポーズ不足のため保持"
        return "unsafeのため保持"
    if effective:
        return "削除（手動編集）"
    return ""


def state_css_class(
    requested: bool,
    effective: bool,
    unsafe_reason: str | None = None,
) -> str:
    """state_label と同じ判定から `.state` に足す状態クラスを返す（純関数）。

    "del"（削除）/ "manual"（削除・手動編集）は枠付きバッジ、"unsafe" は
    現行どおり赤太字テキストのまま（バッジ化しない）。保持は空文字。
    """
    if requested and effective:
        return "del"
    if requested and not effective:
        return "unsafe"
    if effective:
        return "manual"
    return ""


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
        # 行の並び替え（▲▼）。削除指定中の行は TXT 上に行が無く移動が無意味な
        # ため disabled にする（busy 中の無効化はスクリプト側が行う）。
        disabled = " disabled" if seg["requested_delete"] else ""
        parts.append(
            f'<span class="mv">'
            f'<button type="button" class="up" data-i="{seg["index"]}"'
            f' aria-label="この行を上へ"{disabled}>{_ICON_UP}</button>'
            f'<button type="button" class="down" data-i="{seg["index"]}"'
            f' aria-label="この行を下へ"{disabled}>{_ICON_DOWN}</button>'
            f'</span>'
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
        state_cls = state_css_class(
            seg["requested_delete"], seg["effective_delete"], seg.get("unsafe_reason")
        )
        state_classes = f"state {state_cls}" if state_cls else "state"
        label = state_label(
            seg["requested_delete"], seg["effective_delete"], seg["kind"], seg["reason"],
            seg.get("unsafe_reason"),
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
        f'<span class="text">{_ICON_SCISSORS}無音トリム -{trim["removed"]:.1f}秒</span>'
        "</div>"
    )


def _transcript_markup(review: dict) -> str:
    """トランスクリプト全体（表示順に並べ、無音トリムを出力時刻順に差し込む）。

    行は review["order"]（表示順＝TXT の行順）で走査する。segments は json 順
    なので id → セグメントの対応表で引く。無音トリムは出力（編集後）時刻で
    interleave する（並べ替え後は原音時刻の比較では位置がずれるため）。同時刻の
    ときはトリムを先に出す。
    """
    trims = sorted(review["trims"], key=lambda t: t["output"])
    serve = bool(review.get("serve"))
    by_id = {seg["id"]: seg for seg in review["segments"]}
    order = review.get("order") or [seg["id"] for seg in review["segments"]]
    blocks: list[str] = []
    ti = 0
    for seg_id in order:
        seg = by_id.get(seg_id)
        if seg is None:
            continue
        while ti < len(trims) and trims[ti]["output"] <= seg["output"]:
            blocks.append(_trim_markup(trims[ti]))
            ti += 1
        blocks.append(_segment_markup(seg, serve))
    for trim in trims[ti:]:
        blocks.append(_trim_markup(trim))
    return "\n".join(blocks)


# 文字グリフ（✂ / ▲▼）は環境差が大きいため inline SVG に置き換える。色は
# currentColor で親の文字色（トリムの緑・並び替えの muted）へ追従させる。
_ICON_SCISSORS = (
    '<svg class="icon" width="13" height="13" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="1.5" aria-hidden="true">'
    '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>'
    '<path d="M20 4 L8.1 15.9 M14.5 14.5 L20 20 M8.1 8.1 L12 12"/></svg>'
)
# ▲▼ の代替。8px と小さいので stroke は太め（モック準拠）にして視認性を保つ。
_ICON_UP = (
    '<svg class="icon" width="8" height="8" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="3" aria-hidden="true">'
    '<path d="M6 15 L12 9 L18 15"/></svg>'
)
_ICON_DOWN = (
    '<svg class="icon" width="8" height="8" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="3" aria-hidden="true">'
    '<path d="M6 9 L12 15 L18 9"/></svg>'
)


_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=Zen+Old+Mincho:wght@600;700&display=swap">
<style>
/* Flexoki カラースキーム（light/dark）をカスタムプロパティで切替。
   コンポーネント側では hex を直書きしない（配色の変更点をここに集約する）。 */
:root {
  color-scheme: light dark;
  --bg: #FFFCF0;           /* paper */
  --surface: #F2F0E5;      /* base-50 */
  --border: #DAD8CE;       /* base-150 */
  --text: #100F0F;         /* black */
  --text-muted: #6F6E69;   /* base-600 */
  --primary: #1A4F8C;      /* 見出し・アクティブ状態・保存ボタン */
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
  /* Web フォントが読めない環境（オフライン）では後続のフォールバックが効く */
  --font-sans: "Zen Kaku Gothic New", -apple-system, "Hiragino Sans", sans-serif;
  --font-serif: "Zen Old Mincho", "Hiragino Mincho ProN", serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #100F0F;
    --surface: #1C1B1A;    /* base-950 */
    --border: #403E3C;     /* base-800 */
    --text: #CECDC3;       /* base-200 */
    --text-muted: #B7B5AC; /* base-300 */
    --primary: #4385BE;    /* blue-400 */
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
  font-family: var(--font-sans);
  margin: 0 auto; padding: 0 1.5rem 1rem; max-width: 80rem; line-height: 1.8;
  background: var(--bg); color: var(--text);
}
.mono { font-family: var(--font-mono); letter-spacing: 0; }
/* ヘッダー（2行）はスクロールしても画面上部に固定する。背景は不透明にして
   下をスクロールする行が透けないようにする。 */
.header {
  position: sticky; top: 0; z-index: 10; background: var(--bg);
  padding: .75rem 0 .6rem; border-bottom: 1px solid var(--text);
  display: flex; flex-direction: column; gap: .5rem;
}
.hrow { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem .75rem; }
h1.title {
  font-family: var(--font-serif); font-size: 1rem; font-weight: 600;
  color: var(--primary); margin: 0; white-space: nowrap;
}
.subtitle {
  font-family: var(--font-mono); font-size: .7rem; color: var(--text-muted);
  margin: 0;
}
/* アクティブな audio だけを表示するスロット（3要素は DOM に残す）。 */
.transport {
  flex: 1; min-width: 16rem; display: flex; align-items: center;
  border: 1px solid var(--border); border-radius: .15rem; padding: .3rem .5rem;
}
.transport audio { display: none; }
.transport audio.shown { display: block; width: 100%; height: 2rem; }
/* モード切替（原音 / 保存済み）は segmented control。現在案は seg から独立した
   単体ボタン（.mode-solo）で、比較の2択とは見た目のグループを分ける。 */
.seg {
  display: flex; border: 1px solid var(--border); border-radius: .15rem;
  overflow: hidden;
}
.mode {
  font: inherit; font-size: .75rem; padding: .3rem .5rem; cursor: pointer;
  width: 5rem; text-align: center; border: none; border-right: 1px solid var(--border);
  background: transparent; color: var(--text);
}
.mode:last-child { border-right: none; }
.mode.active { background: var(--primary); color: var(--on-accent); }
/* ヘッダー2行目のグループ（比較 / 試聴）とその間の区切り線。 */
.group { display: flex; align-items: center; gap: .4rem; }
.group-divider { align-self: stretch; border-left: 1px solid var(--border); margin: 0 .1rem; }
.mode-solo {
  width: auto; border: 1px solid var(--border); border-radius: .15rem;
}
.mode-solo:disabled { opacity: .5; cursor: default; }
.seek { display: flex; gap: .25rem; }
.seek-btn, .padbtn, .speed-select, .speed-toggle {
  font-family: var(--font-mono); font-size: .7rem; padding: .25rem .5rem;
  cursor: pointer; box-sizing: border-box;
  border: 1px solid var(--border); background: transparent; border-radius: .15rem;
  color: var(--text);
}
.seek-btn:hover { border-color: var(--primary); }
.label { font-size: .75rem; color: var(--text-muted); white-space: nowrap; }
/* 保存ボタン（1行目右端）と XML出力（2行目右端）。通知は各ボタンの左隣。 */
.save-area {
  display: flex; align-items: center; gap: .5rem; margin-left: auto;
}
#save, #export {
  font: inherit; font-size: .8rem; padding: .3rem 1rem; cursor: pointer;
  border-radius: .15rem; box-sizing: border-box; height: 2rem; min-width: 7em;
  border: 1px solid var(--border); background: transparent; color: var(--text);
}
#save:disabled, #export:disabled { opacity: .5; cursor: default; }
#save.dirty {
  background: var(--primary); color: var(--on-accent); border-color: var(--primary);
  font-weight: 700;
}
.msg { font-size: .75rem; color: var(--text-muted); }
.msg.error { color: var(--red); }
/* 再生速度: select（速度値）＋ toggle（OFF x1.0 / ON x選択値）。
   .padbtn は付けない（試聴範囲ボタンの誤反応バグの再発防止のため selector を分離）。 */
.speed-controls, .pad-select {
  display: flex; align-items: center; gap: .25rem; white-space: nowrap;
}
.speed-toggle { width: 3.4em; text-align: center; color: var(--text-muted); }
.speed-toggle.active {
  border-color: var(--text); background: var(--surface); font-weight: 700;
  color: var(--text);
}
.speed-select:disabled, .speed-toggle:disabled { opacity: .5; cursor: default; }
/* 範囲選択はアクション色を使わず、中立色の segmented control にする */
.padbtn.active { border-color: var(--text); background: var(--surface); font-weight: 700; }
.padbtn:disabled { opacity: .5; cursor: default; }
.abs-times {
  font-family: var(--font-mono); font-size: .7rem; color: var(--text-muted);
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
.abs-times b { color: var(--text); font-weight: 400; }
/* ステータスバー（ヘッダー直下・1行）: 現在案の状態ラベル ＋ 統計 */
.statusbar {
  display: flex; align-items: center; gap: .25rem 1.25rem; flex-wrap: wrap;
  padding: .3rem 0; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono); font-size: .7rem; color: var(--text-muted);
}
.statusbar .cur { color: var(--red); font-weight: 500; }
.legend {
  display: flex; flex-wrap: wrap; align-items: center; gap: .25rem 1.25rem;
  font-size: .75rem; color: var(--text-muted);
  margin: 0; padding: .5rem 0; list-style: none;
}
.legend li { display: inline-flex; align-items: center; }
/* 凡例内のサンプルは行レイアウト（flex）を持ち込まず inline で見せる */
.legend .trim, .legend .line {
  display: inline-flex; align-items: center; gap: 0;
  padding: 0; border: none; cursor: default;
}
#transcript { border-top: 1px solid var(--text); }
.line, .trim {
  display: flex; gap: .5rem; align-items: baseline; cursor: pointer;
  padding: .25rem .25rem; border-bottom: 1px solid var(--border);
}
.line:hover, .trim:hover { background: var(--accent-tint-weak); }
/* active 行は左端に 2px の --primary バー（inset box-shadow で行幅を変えない） */
.line.active, .trim.active {
  background: var(--accent-tint-weak); box-shadow: inset 2px 0 0 var(--primary);
}
.ts {
  font-family: var(--font-mono); font-size: .7rem; color: var(--text-muted);
  font-variant-numeric: tabular-nums; min-width: 3.5rem;
}
.sid {
  font-family: var(--font-mono); font-size: .7rem; color: var(--text-muted);
  min-width: 3rem; opacity: .7;
}
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
  /* offset は負にして枠を要素の内側に描く。外側に描くと sticky ヘッダーの
     背景で覆えず、下をスクロールする行の枠線が左右にはみ出て見える */
  outline: 1px dashed var(--red); outline-offset: -1px; background: none;
}
/* 枠が文字に重ならないよう単語には左右の余白を持たせる。凡例サンプルは
   inline-flex＋上下余白で、枠と文字の間隔を実表示と同等に確保する */
.w.unsafe { padding: 0 .2em; }
.legend .w.unsafe { display: inline-block; padding: .1em .4em; }
.trim { color: var(--green); font-size: .8rem; }
.trim .icon { vertical-align: -.15em; margin-right: .25rem; }
.state {
  font-family: var(--font-mono); font-size: .7rem; color: var(--text-muted);
  min-width: 8rem; text-align: right; align-self: center;
}
.state.unsafe { color: var(--red); font-weight: 700; }
/* 「保持」は表示しない（例外だけ目立たせる）。削除／削除（手動編集）は
   行本文の取消線と対で読める中立バッジにする。枠を text 幅まで縮めるため
   min-width を解除し inline-block にする（unsafe は従来どおり全幅のまま）。 */
.state.del, .state.manual {
  display: inline-block; min-width: 0; border: 1px solid var(--text-muted);
  color: var(--text-muted); padding: 0 .3em; border-radius: .15rem;
}
.state.manual { border-color: var(--yellow); color: var(--yellow); }
label.wt { display: inline; }
label.wt input { margin: 0 .1em 0 .2em; vertical-align: middle; }
.line > input.del { margin: 0; align-self: center; accent-color: var(--primary); }
/* 行の並び替えボタン（▲▼相当のSVG）。チェックボックスの隣に置き、行の高さを
   変えないよう小さめにする。削除指定中／busy 中は disabled。 */
.mv { display: flex; gap: .1rem; align-self: center; flex: none; }
.mv button {
  width: 1.15rem; height: 1rem; padding: 0; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); background: transparent;
  border-radius: .15rem; color: var(--text-muted);
}
.mv button:hover:not(:disabled) { border-color: var(--accent); color: var(--text); }
.mv button:disabled { opacity: .4; cursor: default; }
/* キーボード操作のヒント行。長いトランスクリプトでも見えるよう画面下部に
   sticky 固定する（背景は不透明にして下を通る行が透けないようにする）。 */
.kbd {
  position: sticky; bottom: 0; z-index: 10; background: var(--bg);
  display: flex; flex-wrap: wrap; gap: .25rem 1.25rem; margin: 0;
  padding: .4rem 0; border-top: 1px solid var(--border);
  font-family: var(--font-mono); font-size: .625rem; color: var(--text-muted);
}
.kbd b {
  font-weight: 500; color: var(--text); border: 1px solid var(--border);
  border-radius: .15rem; padding: 0 .3rem; background: var(--surface);
}
</style>
</head>
<body>
<header class="header">
<div class="hrow">
  <h1 class="title">$h1</h1>
  <p class="subtitle">$subtitle</p>
  <div class="transport">
    <audio id="audio-src" controls preload="metadata" src="$source_audio" class="player"></audio>
    <audio id="audio-out" controls preload="metadata" src="$review_audio" class="player shown"></audio>
$audio_preview
  </div>
$save_area
</div>
<div class="hrow">
$compare_group  <span class="seek">
    <button type="button" class="seek-btn" data-seek="-10">−10s</button>
    <button type="button" class="seek-btn" data-seek="-5">−5s</button>
    <button type="button" class="seek-btn" data-seek="5">＋5s</button>
    <button type="button" class="seek-btn" data-seek="10">＋10s</button>
  </span>
$transport_extra
$export_area
</div>
</header>
<div class="statusbar">
$status_extra
  <span class="stats">$stats</span>
</div>
<ul class="legend">
  <li>凡例</li>
  <li><span class="w suggest">フィラー提案</span></li>
  <li><span class="w cut">カット済み</span></li>
  <li><span class="line uts">未転写発話候補</span></li>
  <li><span class="w unsafe state">削除指定→安全判定で保持</span></li>
  <li><span class="trim">$scissors_icon無音トリム</span></li>
$click_legend
</ul>
<div id="transcript">
$transcript
</div>
<p class="kbd">$kbd_hints</p>
<script id="sc-data" type="application/json">$data</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("sc-data").textContent);
  // preview は --serve のときだけ存在する（静的モードでは null）。
  var players = {
    src: document.getElementById("audio-src"),
    out: document.getElementById("audio-out"),
    preview: document.getElementById("audio-preview")
  };
  // mode: クリック再生の対象（src / out のみ。既存セマンティクスを保つ）
  // viewMode: ヘッダーのスロットに表示中の audio（src / out / preview）
  var mode = "out";
  var viewMode = "out";
  var stopAt = null;
  var current = null;

  function eachPlayer(fn) {
    Object.keys(players).forEach(function (key) {
      if (players[key]) { fn(players[key], key); }
    });
  }

  function pauseOthers(active) {
    eachPlayer(function (p) { if (p !== active) { p.pause(); } });
  }

  function play(t) {
    var audio = players[mode];
    pauseOthers(audio);
    var from = Math.max(0, t - DATA.pad);
    stopAt = t + DATA.pad;
    current = audio;
    try { audio.currentTime = from; } catch (e) { /* metadata 未読込 */ }
    audio.play();
  }

  eachPlayer(function (p) {
    p.addEventListener("timeupdate", function () {
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

  // モード切替は「ヘッダーに表示する audio」の切替（viewMode）と、
  // 「クリック再生の対象を追従させるか」（syncMode）を分離する。
  // syncMode=true はモードボタンのクリック由来で、src/out のときだけ mode
  // を追従させ、現在案（preview）を選んでも行クリックの再生対象は直前の
  // まま変えない（試聴のトリガは --serve 側）。
  // syncMode=false は再生開始への自動追従（下の play イベント購読）用で、
  // 表示だけを切り替え、mode（src/out の選択）には触れない。
  function setViewMode(next, syncMode) {
    if (!players[next]) { return; }
    viewMode = next;
    if (syncMode && next !== "preview") { mode = next; }
    Array.prototype.forEach.call(
      document.querySelectorAll(".mode"),
      function (b) { b.classList.toggle("active", b.dataset.mode === next); }
    );
    eachPlayer(function (p, key) { p.classList.toggle("shown", key === next); });
  }

  Array.prototype.forEach.call(
    document.querySelectorAll(".mode"),
    function (btn) {
      btn.addEventListener("click", function () { setViewMode(btn.dataset.mode, true); });
    }
  );

  // 再生が始まったプレイヤーへヘッダー表示を自動追従させる（mode は変えない）。
  eachPlayer(function (p, key) {
    p.addEventListener("play", function () { setViewMode(key, false); });
  });

  // シークボタン（表示中のプレイヤーの currentTime を加減、0〜duration にクランプ）
  Array.prototype.forEach.call(
    document.querySelectorAll(".seek-btn"),
    function (btn) {
      btn.addEventListener("click", function () {
        var audio = players[viewMode];
        if (!audio) { return; }
        var dur = isFinite(audio.duration) ? audio.duration : Infinity;
        var next = audio.currentTime + Number(btn.dataset.seek);
        audio.currentTime = Math.min(Math.max(0, next), dur);
      });
    }
  );

  // J/K で active 行を移動して再生する。実際の再生はクリックと同じ経路に
  // 委ねる（--serve では capture 段のハンドラが現在案プレビューへ振り替える）
  // ため、行要素へ合成クリックを送るだけにして分岐を二重に持たない。
  function transcriptRows() {
    return document.querySelectorAll("#transcript .line, #transcript .trim");
  }

  function moveActiveRow(delta) {
    var rows = transcriptRows();
    if (!rows.length) { return; }
    var cur = document.querySelector("#transcript .active");
    var index = Array.prototype.indexOf.call(rows, cur);
    var next = index < 0
      ? (delta > 0 ? 0 : rows.length - 1)
      : Math.min(Math.max(index + delta, 0), rows.length - 1);
    var row = rows[next];
    row.scrollIntoView({ block: "nearest" });
    row.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  }

  // 入力要素（フォーム部品・audio・contenteditable）にフォーカスがある間は
  // キーボードショートカットを発火させない。
  function isTypingTarget(ev) {
    return !!(
      ev.target && ev.target.closest &&
      ev.target.closest("input, button, select, textarea, audio, [contenteditable]")
    );
  }

  // H/L は表示モード切替（原音/保存済み/現在案を並び順の系列とみなし、
  // vim 風に1つ左/右へ移動）。setViewMode を直接呼ばず対応する .mode
  // ボタンへ合成クリックを送ることで、disabled の尊重や active 表示の
  // 更新をクリック時と同じ経路に一本化する。disabled のボタン（初回試聴前
  // の現在案）は候補から除外し、存在しないボタン（静的モードの現在案）は
  // querySelectorAll に現れないため自然に除外される。端では止まる
  // （ラップしない）。
  function moveMode(delta) {
    var buttons = Array.prototype.filter.call(
      document.querySelectorAll(".mode"),
      function (b) { return !b.disabled; }
    );
    if (!buttons.length) { return; }
    var cur = document.querySelector(".mode.active");
    var index = Array.prototype.indexOf.call(buttons, cur);
    var next = index < 0
      ? (delta > 0 ? 0 : buttons.length - 1)
      : Math.min(Math.max(index + delta, 0), buttons.length - 1);
    buttons[next].click();
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey || ev.repeat) { return; }
    if (isTypingTarget(ev)) { return; }
    if (ev.code === "Space") {
      var audio = players[viewMode];
      // 表示中が現在案 preview で一度も試聴していなければ何もしない（再生する音が無い）。
      if (viewMode === "preview" && !audio.currentSrc) { return; }
      ev.preventDefault();
      if (audio.paused) {
        audio.play().catch(function () { /* 再生開始前の中断は無視する */ });
      } else {
        audio.pause();
      }
      return;
    }
    var key = (ev.key || "").toLowerCase();
    if (key === "h" || key === "l") {
      ev.preventDefault();
      moveMode(key === "h" ? -1 : 1);
      return;
    }
    if (key !== "j" && key !== "k") { return; }
    ev.preventDefault();
    moveActiveRow(key === "j" ? 1 : -1);
  });
})();
</script>
$serve_script
</body>
</html>
""")


# --serve 時だけ差し込むヘッダー1行目右端の保存ボタン（通知はボタンの左隣）
_SAVE_AREA = Template("""  <span class="save-area">
    <span class="msg" id="save-msg"></span>
    <button type="button" id="save" disabled title="保存と同時に .logic.fcpxml も書き出します">保存済み</button>
  </span>""")

# --serve 時だけ差し込むヘッダー2行目の中央部（速度・絶対時刻）。試聴範囲
# （前後 padbtn）は比較/試聴グループ側（_COMPARE_GROUP_SERVE）に属するため
# ここには含めない。配置順は 速度→絶対時刻 で固定する。
_TRANSPORT_EXTRA = Template("""  <span class="speed-controls" id="speed-controls">
    <span class="label">速度</span>
    <select class="speed-select" id="speed-select">
      <option value="1.25">x1.25</option>
      <option value="1.5" selected>x1.5</option>
      <option value="1.75">x1.75</option>
      <option value="2">x2.0</option>
    </select>
    <button type="button" class="speed-toggle" id="speed-toggle">OFF</button>
  </span>
  <span class="abs-times">対象 原音 <b id="abs-target">--:--.-</b> → 現在案 <b id="abs-center">--:--.-</b></span>""")

# --serve 時だけ差し込むヘッダー2行目右端の FCPXML 書き出しボタン。
# 書き出し結果の通知はボタンの左隣に出し、成功時は自動で消す（保存と同じ規則）。
_EXPORT_AREA = Template("""  <span class="save-area">
    <span class="msg" id="export-msg"></span>
    <button type="button" id="export" title="$export_title">XML出力</button>
  </span>""")

# --serve 時だけ差し込むステータスバー左側（現在案の状態ラベルとエラー通知）。
_STATUS_EXTRA = """  <span class="cur" id="preview-state">現在案：未選択</span>
  <span class="msg" id="preview-msg"></span>"""

# --serve 時だけヘッダーへ足す audio（現在案）。
_AUDIO_PREVIEW = (
    '    <audio id="audio-preview" controls preload="none" class="player"></audio>'
)

# ヘッダー2行目の左側（比較 / 試聴）。静的モードは「比較」の2択のみで見出しは
# 付けない。--serve は「比較」（原音/保存済み）と「試聴」（現在案 ＋ 前後）を
# ヘアラインで分け、意味の非対称（比較の2択 / 未保存編集の試聴）を構成で示す。
_COMPARE_GROUP_STATIC = Template("""  <span class="group">
    <span class="seg">
      <button type="button" class="mode" data-mode="src">原音</button>
      <button type="button" class="mode active" data-mode="out">保存済み</button>
    </span>
  </span>
""")
_COMPARE_GROUP_SERVE = Template("""  <span class="group">
    <span class="label">比較</span>
    <span class="seg">
      <button type="button" class="mode" data-mode="src">原音</button>
      <button type="button" class="mode active" data-mode="out">保存済み</button>
    </span>
  </span>
  <span class="group-divider"></span>
  <span class="group">
    <span class="label">試聴</span>
    <button type="button" class="mode mode-solo" id="mode-preview" data-mode="preview"
      disabled title="編集中（未保存）の現在案を前後のみ試聴">現在案</button>
    <span class="pad-select">
      <span class="label">前後</span>
      <button type="button" class="padbtn" data-pad="2">2秒</button>
      <button type="button" class="padbtn active" data-pad="4">4秒</button>
      <button type="button" class="padbtn" data-pad="8">8秒</button>
    </span>
  </span>
""")

# キーボード操作ヒント（serve 専用キーは serve のときだけ出す）。
_KBD_MOVE = "<span><b>J</b> / <b>K</b> 行移動・再生</span>"
_KBD_HINTS_STATIC = (
    "<span><b>Space</b> 再生 / 停止</span>"
    "<span><b>H</b> / <b>L</b> 表示切替</span>"
    + _KBD_MOVE
)
_KBD_HINTS_SERVE = (
    "<span><b>Space</b> 再生 / 停止</span>"
    "<span><b>H</b> / <b>L</b> 表示切替</span>"
    + _KBD_MOVE
    + "<span><b>X</b> 削除トグル</span>"
    "<span><b>P</b> 現在案を試聴</span>"
    "<span><b>S</b> 保存</span>"
    "<span><b>O</b> 保存済みをその位置から</span>"
    "<span><b>Option＋クリック</b> 保存済みをクリック位置から再生</span>"
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
  var previewModeBtn = document.getElementById("mode-preview");
  var saveBtn = document.getElementById("save");
  var exportBtn = document.getElementById("export");
  var saveMsg = document.getElementById("save-msg");
  var previewMsg = document.getElementById("preview-msg");
  var previewState = document.getElementById("preview-state");
  var absTarget = document.getElementById("abs-target");
  var absCenter = document.getElementById("abs-center");
  var dirty = false;
  var busy = false;
  // busy 中の種別。null（非busy）/ "preview"（試聴生成中） / "save"（保存中） /
  // "export"（FCPXML書き出し中）。preview 中は保存/書き出しボタンの文言を
  // 維持したまま disable し、save/export 中だけ専用の進行中文言に切り替える。
  var busyKind = null;
  var baseline = snapshot();
  // 3段目「現在案プレビュー」の試聴範囲（秒）。3段目付近のボタンで切替できる。
  var currentPad = 4;
  // 直前にプレビューした対象（行/単語）。試聴範囲を切り替えたときの再生成に使う。
  var currentTarget = null;
  // トグル直後は自動再生せず、取得した音声をキャッシュして
  // 対象の行/単語をクリックしたときに再生する（key -> {url, ...}）。
  // キーには pad を含める（対象・pad・編集状態が一致する場合のみ再利用する）。
  var previewCache = {};
  // /preview, /export の応答が世代遅れ（その後に新しい操作が始まっている）なら
  // 状態を巻き戻さないよう、リクエストごとに世代IDを振って照合する。
  var generation = 0;

  var allPlayers = [audioSrc, audioOut, preview];

  // 排他再生。いずれかの audio が再生開始したら他2つを止める（3つの audio が
  // 同時に鳴らないようにする）。
  function pauseOtherPlayers(active) {
    allPlayers.forEach(function (other) {
      if (other !== active) { other.pause(); }
    });
  }

  allPlayers.forEach(function (player) {
    player.addEventListener("play", function () { pauseOtherPlayers(player); });
  });

  // 再生速度（select で選択した値 ＋ OFF/ON toggle）。3プレイヤー全てに適用する。
  // audio-preview は src を差し替えるたび playbackRate がリセットされるため、
  // 単純に一度適用するだけでは足りない。各 audio の "play"／"loadedmetadata"
  // イベントで現在速度を再適用し、src 差し替え後の再生でも速度を保つ。
  //
  // 状態モデル: selectedSpeed（select の値）／speedEnabled（toggle の状態）／
  // currentSpeed = speedEnabled ? selectedSpeed : 1。速度ハンドラは
  // playbackRate と表示更新だけを行い、currentPad/currentTarget・
  // play/pause/load/src/currentTime・試聴範囲の active には一切触れない
  // （padbtn を共有していた旧速度ボタンが試聴範囲ハンドラに誤反応していた
  // バグの再発防止のため、クラスも selector も完全に分離する）。
  var speedSelect = document.getElementById("speed-select");
  var speedToggle = document.getElementById("speed-toggle");
  var selectedSpeed = 1.5;
  var speedEnabled = false;
  var currentSpeed = 1;

  function applyPlaybackSpeed() {
    allPlayers.forEach(function (p) { p.playbackRate = currentSpeed; });
  }

  allPlayers.forEach(function (player) {
    player.addEventListener("play", applyPlaybackSpeed);
    player.addEventListener("loadedmetadata", applyPlaybackSpeed);
  });

  function updateSpeedToggleLabel() {
    // 表示は ON/OFF（select と x 値が2つ並ぶと紛らわしいため。値は select が示す）
    speedToggle.textContent = speedEnabled ? "ON" : "OFF";
    speedToggle.classList.toggle("active", speedEnabled);
  }

  speedSelect.addEventListener("change", function () {
    selectedSpeed = Number(speedSelect.value);
    // OFF 中の select 変更は選択値の更新のみ（適用しない）。ON 中は即適用する。
    if (speedEnabled) {
      currentSpeed = selectedSpeed;
      applyPlaybackSpeed();
    }
    updateSpeedToggleLabel();
  });

  speedToggle.addEventListener("click", function () {
    speedEnabled = !speedEnabled;
    currentSpeed = speedEnabled ? selectedSpeed : 1;
    updateSpeedToggleLabel();
    applyPlaybackSpeed();
  });

  // 行/単語の対象（{type, id, word}）から savedPositions / previewCache の
  // キー（L<id> / W<id>:<word>）を作る。
  function targetKey(target) {
    return target.type === "word"
      ? "W" + target.id + ":" + target.word
      : "L" + target.id;
  }

  function previewKey(target, pad) {
    return targetKey(target) + "@" + pad;
  }

  // json 順の DATA.segments から id で1件探す。
  function segmentById(id) {
    var found = null;
    DATA.segments.forEach(function (s) { if (s.id === id) { found = s; } });
    return found;
  }

  // 行要素（.line）に対応する json セグメント（data-i は json 順の index）。
  function segmentOfLine(el) {
    return el ? DATA.segments[Number(el.dataset.i)] : null;
  }

  // その行が削除指定中か（並び替えの可否・スワップ相手の判定に使う）。
  function rowDeleted(el) {
    var seg = segmentOfLine(el);
    return !!(seg && seg.requested_delete);
  }

  // DOM の行順から DATA.order（全 ID の順列）を作り直す。トリム擬似行は
  // TXT の行ではないので order には含めない。
  function rebuildOrder() {
    var ids = [];
    Array.prototype.forEach.call(
      document.querySelectorAll("#transcript .line"),
      function (el) {
        var seg = segmentOfLine(el);
        if (seg) { ids.push(seg.id); }
      }
    );
    DATA.order = ids;
  }

  // 入れ替え相手の隣接行。トリム擬似行と削除指定中の行は飛ばす（TXT に行が
  // 無いものと入れ替えても意味が無いため）。端まで到達したら null（no-op）。
  function swapSibling(line, dir) {
    var node = dir < 0 ? line.previousElementSibling : line.nextElementSibling;
    while (node) {
      if (node.classList.contains("line") && !rowDeleted(node)) { return node; }
      node = dir < 0 ? node.previousElementSibling : node.nextElementSibling;
    }
    return null;
  }

  // 行を1つ上（dir<0）／下（dir>0）へ動かす。自動プレビューはしない
  // （▲▼は連打されるため。保存/クリック時に現在の順序で作り直される）。
  function moveLine(line, dir) {
    var other = swapSibling(line, dir);
    if (!other) { return; }
    var parent = line.parentNode;
    if (dir < 0) {
      parent.insertBefore(line, other);
    } else {
      parent.insertBefore(other, line);
    }
    rebuildOrder();
    updateSaveButton();
    clearPreviewCache();
  }

  function snapshot() {
    var s = {};
    DATA.segments.forEach(function (seg) {
      s["L" + seg.id] = seg.requested_delete;
      seg.words.forEach(function (w) {
        if (w.suggest) { s["W" + seg.id + ":" + w.index] = w.requested_delete; }
      });
    });
    // 並び順は1つのキーにまとめる（何行動かしても dirty 件数は1件）。
    s["O"] = DATA.order.join("\\n");
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

  // 保存ボタンの表示を save中／preview中／変更あり（件数入り）／変更なしの
  // 4状態で切り替える。preview中は「保存中…」ではなく通常の文言を維持した
  // まま disable する（保存していないのに保存中と誤表示しないため）。
  function updateSaveButton() {
    if (busyKind === "save") {
      saveBtn.disabled = true;
      saveBtn.classList.remove("dirty");
      saveBtn.textContent = "保存中…";
      return;
    }
    var n = dirtyKeys().length;
    dirty = n > 0;
    saveBtn.disabled = !dirty || busy;
    saveBtn.classList.toggle("dirty", dirty && !busy);
    saveBtn.textContent = dirty ? "変更を保存（" + n + "件）" : "保存済み";
  }

  // FCPXML書き出しボタンの表示を export中／通常の2状態で切り替える。dirty で
  // なくても常時利用可（busy 中を除く）で、export 中だけ「書き出し中…」にする。
  function updateExportButton() {
    if (busyKind === "export") {
      exportBtn.disabled = true;
      exportBtn.textContent = "書き出し中…";
      return;
    }
    exportBtn.disabled = busy;
    exportBtn.textContent = "XML出力";
  }

  // busy 中は全チェックボックスと保存/書き出しボタン・pad切替ボタン
  // （.padbtn[data-pad]）・速度コントロール（.speed-select/.speed-toggle）を
  // 操作不能にする（見た目にも分かるようにする）。busy を素通しして黙って
  // 古い操作を捨てる経路を作らないため。kind は "preview" / "save" /
  // "export" / 省略（解除時）。
  function setControlsBusy(value, kind) {
    busy = value;
    busyKind = value ? (kind || null) : null;
    Array.prototype.forEach.call(
      document.querySelectorAll("input.del, input.delw"),
      function (box) { box.disabled = value; }
    );
    Array.prototype.forEach.call(
      document.querySelectorAll(".padbtn[data-pad]"),
      function (btn) { btn.disabled = value; }
    );
    Array.prototype.forEach.call(
      document.querySelectorAll(".speed-select, .speed-toggle"),
      function (el) { el.disabled = value; }
    );
    // 並び替えボタンは busy 解除時に一律 enable してはいけない（削除指定中の
    // 行は常に disabled のまま）。busy と行の削除指定の論理和で決める。
    Array.prototype.forEach.call(
      document.querySelectorAll(".mv button"),
      function (btn) {
        var line = btn.closest(".line");
        btn.disabled = value || (line ? rowDeleted(line) : false);
      }
    );
    updateSaveButton();
    updateExportButton();
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

  // 3段目左ラベル（現在案の状態）。preview応答の message をそのまま表示する。
  function noteState(text) {
    previewState.textContent = text;
  }

  // previewエラーは状態ラベルとは別要素へ出し、混ぜない。
  function notePreviewError(text) {
    previewMsg.textContent = text;
    previewMsg.classList.add("error");
  }

  function clearPreviewError() {
    previewMsg.textContent = "";
    previewMsg.classList.remove("error");
  }

  // 保存・書き出しの通知は同じ振る舞い: 各ボタンの左隣に出し、成功メッセージは
  // 数秒で自動的に消す（エラーは操作されるまで残す）。
  function makeNotice(el) {
    var timer = null;
    return function (text, isError) {
      if (timer) { clearTimeout(timer); timer = null; }
      el.textContent = text;
      el.classList.toggle("error", !!isError);
      if (!isError) {
        timer = setTimeout(function () { el.textContent = ""; }, 5000);
      }
    };
  }
  var noteSave = makeNotice(saveMsg);
  var noteExport = makeNotice(document.getElementById("export-msg"));

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

  // キャッシュ済み/応答直後の試聴データから「対象 原音」「現在案」の絶対時刻表示を更新する。
  function updateAbsTimes(entry) {
    absTarget.textContent = formatTime(entry.target);
    absCenter.textContent = formatTime(entry.center);
  }

  function stateLabel(requested, effective, unsafeReason) {
    if (requested && effective) { return "削除"; }
    if (requested) {
      return unsafeReason === "pause" ? "ポーズ不足のため保持" : "unsafeのため保持";
    }
    if (effective) { return "削除（手動編集）"; }
    return ""; // 保持は例外ではないため表示しない
  }

  // state_css_class（Python 側）と同じ判定。del/manual は枠付きバッジ、
  // unsafe は赤太字テキストのまま（バッジ化しない）。
  function stateClass(requested, effective) {
    if (requested && effective) { return "del"; }
    if (requested) { return "unsafe"; }
    if (effective) { return "manual"; }
    return "";
  }

  function applyStates() {
    DATA.segments.forEach(function (seg) {
      var line = document.querySelector('.line[data-i="' + seg.index + '"]');
      if (!line) { return; }
      var cls = stateClass(seg.requested_delete, seg.effective_delete);
      var isUnsafe = cls === "unsafe";
      line.classList.toggle("deleted", seg.effective_delete);
      line.classList.toggle("unsafe", isUnsafe);
      var box = line.querySelector("input.del");
      if (box) { box.checked = seg.requested_delete; }
      // 削除指定中の行は並び替えできない（busy 中も操作させない）。
      Array.prototype.forEach.call(
        line.querySelectorAll(".mv button"),
        function (btn) { btn.disabled = busy || seg.requested_delete; }
      );
      var state = line.querySelector(".state");
      if (state) {
        state.textContent = stateLabel(
          seg.requested_delete, seg.effective_delete, seg.unsafe_reason
        );
        state.classList.toggle("unsafe", cls === "unsafe");
        state.classList.toggle("del", cls === "del");
        state.classList.toggle("manual", cls === "manual");
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
          // エラー時も payload 全体を保持する（txt_saved/review_updated/
          // segments 等から呼び出し側が画面同期を判断できるようにするため）。
          var err = new Error(payload.error || ("HTTP " + res.status));
          err.payload = payload;
          throw err;
        }
        return payload;
      });
    });
  }

  // パスの basename（フルパスでなく表示用のファイル名だけを見せるため）。
  function basename(path) {
    var parts = String(path).split(/[\\/]/);
    return parts[parts.length - 1];
  }

  function findWord(seg, index) {
    var found = null;
    seg.words.forEach(function (w) { if (w.index === index) { found = w; } });
    return found;
  }

  // 保存済み位置表（L<id> / W<id>:<word> -> {sourceCenter, outputCenter, outputStart}）。
  // 保存（/export）成功時（mergeSavedStates）だけ再構築する。preview の応答では
  // 触らない（未保存の試聴で保存済み位置がずれて見えることを防ぐ）。
  var savedPositions = {};

  function capturePositions(segments) {
    savedPositions = {};
    segments.forEach(function (seg) {
      savedPositions["L" + seg.id] = {
        sourceCenter: seg.source_center,
        outputCenter: seg.output_center,
        outputStart: seg.output
      };
      seg.words.forEach(function (w) {
        savedPositions["W" + seg.id + ":" + w.index] = {
          sourceCenter: w.center,
          outputCenter: w.output_center,
          outputStart: w.output
        };
      });
    });
  }

  // /preview 成功時に使う。ユーザーが指定した削除状態（requested_delete）と
  // 保存済み位置表（savedPositions）は一切変更せず、現在案の反映結果
  // （effective_delete と現在案の位置 output_center）だけを更新する。
  function mergePreviewStates(segments) {
    var byId = {};
    segments.forEach(function (s) { byId[s.id] = s; });
    DATA.segments.forEach(function (seg) {
      var fresh = byId[seg.id];
      if (!fresh) { return; }
      seg.effective_delete = fresh.effective_delete;
      seg.unsafe_reason = fresh.unsafe_reason;
      seg.output_center = fresh.output_center;
      seg.words.forEach(function (w) {
        var fw = findWord(fresh, w.index);
        if (!fw) { return; }
        w.effective_delete = fw.effective_delete;
        w.output_center = fw.output_center;
      });
    });
  }

  // 保存（/export）成功時だけ使う。全状態を確定させ、保存済み位置表を再構築する。
  function mergeSavedStates(segments) {
    var byId = {};
    segments.forEach(function (s) { byId[s.id] = s; });
    DATA.segments.forEach(function (seg) {
      var fresh = byId[seg.id];
      if (!fresh) { return; }
      seg.requested_delete = fresh.requested_delete;
      seg.effective_delete = fresh.effective_delete;
      seg.unsafe_reason = fresh.unsafe_reason;
      seg.state = fresh.state;
      seg.output = fresh.output;
      seg.source_center = fresh.source_center;
      seg.output_center = fresh.output_center;
      seg.words.forEach(function (w) {
        var fw = findWord(fresh, w.index);
        if (!fw) { return; }
        w.requested_delete = fw.requested_delete;
        w.effective_delete = fw.effective_delete;
        w.output = fw.output;
        w.center = fw.center;
        w.output_center = fw.output_center;
      });
    });
    capturePositions(DATA.segments);
  }

  // 保存ボタン・XML出力ボタン（共に /export）の応答を画面へ反映する共通経路。txt_saved（保存 baseline
  // の基準）と review_updated（review.wav 再生成の成否）は独立した2軸のため、
  // それぞれ opts.commitBaseline と payload.review_updated の厳密比較
  // （=== true）で個別に判定する。segments が無い応答（review.wav 再生成
  // 失敗など）では音源・データの同期をスキップする。
  function applyReviewPayload(payload, opts) {
    var commitBaseline = !!(opts && opts.commitBaseline);
    if (payload.review_updated === true) {
      var out = document.getElementById("audio-out");
      out.src = DATA.review_audio + "&v=" + payload.version;
      out.load();
      clearPreviewCache();
      if (payload.segments) {
        mergeSavedStates(payload.segments);
        applyStates();
        if (payload.stats) {
          document.querySelector(".stats").textContent = payload.stats.join(" · ");
        }
      }
    }
    if (commitBaseline) {
      baseline = snapshot();
    }
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
    // currentPad はボタン操作でいつでも変わるため、この試聴リクエスト専用の
    // 値としてここで固定する。以降（POST body・キャッシュキー・応答処理）は
    // すべて requestPad を使い、応答処理中にグローバル currentPad を
    // 参照しない（範囲変更中の競合・誤キャッシュを防ぐ）。
    var requestPad = currentPad;
    var myGeneration = ++generation;
    setControlsBusy(true, "preview");
    noteState("試聴を生成中…");
    clearPreviewError();
    post("/preview", {
      toggles: toggles(), order: DATA.order, target: target, pad_s: requestPad
    })
      .then(function (payload) {
        // この応答より後に新しい操作（トグル/保存）が始まっていたら、
        // 画面の状態を巻き戻さないよう破棄する。
        if (myGeneration !== generation) { return; }
        mergePreviewStates(payload.segments);
        applyStates();
        noteState(payload.message || "現在案");
        clearPreviewError();
        // 対象の行/単語をクリックしたときにも再生できるよう音声をキャッシュ
        // しておく（従来どおり）。加えて、チェック変更の結果をすぐ耳で確認
        // できるよう audio-preview へも即座に設定し load/play を試みる
        // （autoplay 拒否は握りつぶし、src 設定と load だけは完了させる）。
        var key = previewKey(target, requestPad);
        var url = URL.createObjectURL(toBlob(payload.audio));
        var entry = {
          url: url,
          center: payload.center_s,
          target: payload.target_source_s,
          pad: requestPad
        };
        previewCache[key] = entry;
        updateAbsTimes(entry);
        preview.src = url;
        preview.load();
        // 初回の試聴生成が完了するまで「現在案」ボタンは disabled のまま
        // （src が無い preview を選ばせないため）。以降は毎回 no-op。
        previewModeBtn.disabled = false;
        var playing = preview.play();
        if (playing && playing.catch) { playing.catch(function () {}); }
      })
      .catch(function (err) {
        if (myGeneration !== generation) { return; }
        notePreviewError(err.message);
      })
      .then(function () {
        if (myGeneration !== generation) { return; }
        setControlsBusy(false);
      });
  }

  // 現在案プレビューを再生する（対象＋現在の試聴範囲でキャッシュ済みなら即再生、
  // 無ければ現在のチェック状態で /preview を生成してから再生する）。
  function playCurrentPlan(target) {
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

  // 比較窓（現在案 preview と同じ幅）の開始位置。center - pad を 0 未満に
  // させない。
  function comparisonStart(center, pad) {
    return Math.max(0, center - pad);
  }

  // 原音・保存済みプレイヤーを対象の比較窓開始位置へシークする（play() は
  // 呼ばない。実際の再生は現在案 preview だけが行う）。原音は対象そのものの
  // source_center（保存の有無に関わらず不変）を使い、保存済みは
  // savedPositions（保存＝/export 成功時にしか更新されない）の output_center だけを
  // 使う。未保存の現在案 output を保存済みプレイヤーのシークに使わない。
  function primeComparisonPlayers(target, pad) {
    var seg = segmentById(target.id);
    if (!seg) { return; }
    var wordData = target.type === "word" ? findWord(seg, target.word) : null;
    var sourceCenter = wordData ? wordData.center : seg.source_center;
    try {
      audioSrc.currentTime = comparisonStart(sourceCenter, pad);
    } catch (e) { /* metadata 未読込 */ }
    var saved = savedPositions[targetKey(target)];
    if (saved) {
      try {
        audioOut.currentTime = comparisonStart(saved.outputCenter, pad);
      } catch (e) { /* metadata 未読込 */ }
    }
  }

  // 対象に保存済み位置があるかどうか（Option+クリック再生の可否判定に使う）。
  function savedPositionFor(target) {
    return savedPositions[targetKey(target)] || null;
  }

  // 保存済み（review.wav）を対象の開始位置ちょうどから再生する。pad 秒手前
  // ではなく対象そのものの開始位置（savedPositions[...].outputStart、既存の
  // seg.output / w.output をそのまま使う）へシークする点が primeComparisonPlayers
  // との違い。Option+クリックと O キーの両方から呼ばれる共通経路。
  function playSavedFrom(target) {
    var saved = savedPositionFor(target);
    if (!saved) {
      notePreviewError("保存済み位置が不明です");
      return;
    }
    try {
      audioOut.currentTime = saved.outputStart;
    } catch (e) { /* metadata 未読込 */ }
    var playing = audioOut.play();
    if (playing && playing.catch) { playing.catch(function () {}); }
  }

  function setActiveLine(el) {
    var prev = document.querySelector(".line.active, .trim.active");
    if (prev) { prev.classList.remove("active"); }
    el.classList.add("active");
  }

  // active 行表示と currentTarget（範囲2/4/8秒変更時の再プライム対象）の更新を
  // 一箇所にまとめる。通常クリック・Option+クリックの両経路がここを1回だけ通る。
  function selectTarget(line, target) {
    setActiveLine(line);
    currentTarget = target;
  }

  // 行/単語クリックの再生先を切り替える（capture フェーズで先取りし、本文
  // スクリプト側の既定動作＝上段2プレイヤーへは渡さない）。
  // 通常クリック＝3段目「現在案プレビュー」。無音トリムは対象が無いため上段
  // プレイヤーの既定動作（本文スクリプト側）に委ねる。
  document.getElementById("transcript").addEventListener("click", function (ev) {
    // 並び替え（▲▼）は再生を伴わない操作なので、行クリックの再生経路へは
    // 一切渡さない（本文スクリプト側の既定動作も止める）。
    var mvBtn = ev.target.closest(".mv button");
    if (mvBtn) {
      ev.stopPropagation();
      if (busy || mvBtn.disabled) { return; }
      moveLine(mvBtn.closest(".line"), mvBtn.classList.contains("up") ? -1 : 1);
      return;
    }
    if (ev.target.closest("input, label.wt")) { return; }
    if (ev.target.closest(".trim")) { return; }
    var line = ev.target.closest(".line");
    if (!line) { return; }
    var seg = DATA.segments[Number(line.dataset.i)];
    if (!seg) { return; }
    var word = ev.target.closest(".w");
    var wordData = word ? findWord(seg, Number(word.dataset.w)) : null;

    ev.stopPropagation();
    // busy 中は対象を変更しない（active 表示・試聴対象を黙って切り替えない）。
    // 生成中であることをその場で伝えるだけにとどめる。
    if (busy) {
      noteState("試聴を生成中です");
      return;
    }

    var target = wordData
      ? { type: "word", id: seg.id, word: wordData.index }
      : { type: "line", id: seg.id };
    selectTarget(line, target);
    primeComparisonPlayers(target, currentPad);
    if (ev.altKey) {
      ev.preventDefault();          // Option(Alt)+click には非リンク要素で横取りすべき
                                     // ブラウザ既定動作は無いが、他の修飾キー分岐と
                                     // 挙動を揃えるため一貫して呼んでおく。
      playSavedFrom(target);
      return;                       // preview へは進まない
    }
    playCurrentPlan(target);
  }, true);

  // 試聴範囲（±2/±4/±8秒）の切替。現在選択中の対象があれば再生成する。
  // selector は .padbtn[data-pad] に限定する（速度UIは .padbtn を持たないが、
  // 万一クラスが共有されても誤反応しないよう data-pad の有無で明示的に絞る）。
  Array.prototype.forEach.call(
    document.querySelectorAll(".padbtn[data-pad]"),
    function (btn) {
      btn.addEventListener("click", function () {
        var pad = Number(btn.dataset.pad);
        if (pad === currentPad) { return; }
        currentPad = pad;
        Array.prototype.forEach.call(
          document.querySelectorAll(".padbtn[data-pad]"),
          function (b) { b.classList.remove("active"); }
        );
        btn.classList.add("active");
        if (currentTarget) {
          primeComparisonPlayers(currentTarget, pad);
          playCurrentPlan(currentTarget);
        }
      });
    }
  );

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
    setControlsBusy(true, "save");
    post("/export", { toggles: toggles(), order: DATA.order })
      .then(function (payload) {
        if (myGeneration !== generation) { return; }
        applyReviewPayload(payload, { commitBaseline: payload.txt_saved === true });
        noteSave("保存しました（XML出力済み）: " + basename(payload.fcpxml_path), false);
      })
      .catch(function (err) {
        if (myGeneration !== generation) { return; }
        var payload = err.payload || {};
        applyReviewPayload(payload, { commitBaseline: payload.txt_saved === true });
        noteSave(err.message, true);
      })
      .then(function () {
        if (myGeneration !== generation) { return; }
        setControlsBusy(false);
      });
  });

  exportBtn.addEventListener("click", function () {
    if (busy) { return; }
    var myGeneration = ++generation;
    setControlsBusy(true, "export");
    post("/export", { toggles: toggles(), order: DATA.order })
      .then(function (payload) {
        if (myGeneration !== generation) { return; }
        applyReviewPayload(payload, { commitBaseline: payload.txt_saved === true });
        noteExport("書き出し済み: " + basename(payload.fcpxml_path), false);
      })
      .catch(function (err) {
        if (myGeneration !== generation) { return; }
        var payload = err.payload || {};
        applyReviewPayload(payload, { commitBaseline: payload.txt_saved === true });
        noteExport(err.message, true);
      })
      .then(function () {
        if (myGeneration !== generation) { return; }
        setControlsBusy(false);
      });
  });

  // active 行（トリム擬似行は対象外）の削除チェックボックスを反転する。
  // change を明示的に発火させ、クリック操作と同じ経路（試聴の再生成・保存
  // ボタン更新）を通す。busy 中と disabled のチェックボックスは無視する。
  function toggleActiveLineDelete() {
    var line = document.querySelector("#transcript .line.active");
    if (!line || busy) { return; }
    var box = line.querySelector("input.del");
    if (!box || box.disabled) { return; }
    box.checked = !box.checked;
    box.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // 入力要素にフォーカスがある間はキーボードショートカットを発火させない。
  function isTypingTarget(ev) {
    return !!(
      ev.target && ev.target.closest &&
      ev.target.closest("input, button, select, textarea, audio, [contenteditable]")
    );
  }

  // X/P/S/O は --serve 専用の操作（削除トグル／現在案を試聴／保存／保存済みを
  // その位置から再生）。J/K（行移動）・Space（再生/停止）は本文スクリプト側が持つ。
  // 修飾キー・キーリピート中は横取りしない。
  document.addEventListener("keydown", function (ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey || ev.repeat) { return; }
    if (isTypingTarget(ev)) { return; }
    var key = (ev.key || "").toLowerCase();
    if (key === "x") {
      ev.preventDefault();
      toggleActiveLineDelete();
    } else if (key === "p") {
      ev.preventDefault();
      if (busy || !currentTarget) { return; }
      playCurrentPlan(currentTarget);
    } else if (key === "s") {
      ev.preventDefault();
      if (saveBtn.disabled) { return; }
      saveBtn.click();
    } else if (key === "o") {
      ev.preventDefault();
      if (!currentTarget) {
        notePreviewError("対象が選択されていません");
        return;
      }
      playSavedFrom(currentTarget);
    }
  });

  capturePositions(DATA.segments);
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
    # serve（Hlyst 対話モード）はクリック＝現在案 preview 専用のため、静的モード専用の
    # 固定 pad 秒クリック説明はここでは出さない（serve モードへ固定の「前後2秒」を
    # 表示しない）。
    # serve モードの Option+クリック説明は下部キーボードヒントへ移した（重複表示を避ける）。
    click_legend = (
        ""
        if serve
        else f'<li>行・単語をクリックすると前後 {review["pad"]:g} 秒を再生します</li>'
    )
    save_area = _SAVE_AREA.substitute() if serve else ""
    export_area = (
        _EXPORT_AREA.substitute(export_title=escape(EXPORT_NOTE, quote=True))
        if serve
        else ""
    )
    return _HTML_TEMPLATE.substitute(
        title=escape(f"Hlyst — {review['base']}"),
        h1=escape(HLYST_H1),
        subtitle=escape(HLYST_SUBTITLE),
        save_area=save_area,
        export_area=export_area,
        transport_extra=_TRANSPORT_EXTRA.substitute() if serve else "",
        status_extra=_STATUS_EXTRA if serve else "",
        audio_preview=_AUDIO_PREVIEW if serve else "",
        compare_group=(
            _COMPARE_GROUP_SERVE.substitute() if serve else _COMPARE_GROUP_STATIC.substitute()
        ),
        kbd_hints=_KBD_HINTS_SERVE if serve else _KBD_HINTS_STATIC,
        scissors_icon=_ICON_SCISSORS,
        click_legend=click_legend,
        serve_script=_SERVE_SCRIPT if serve else "",
        source_audio=escape(review["source_audio"], quote=True),
        review_audio=escape(review["review_audio"], quote=True),
        # ステータスバーは1行のため、複数行の統計を中黒で連結して1行にする
        stats=escape(" · ".join(review["stats"])),
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
    kept_order = [
        seg_id
        for seg_id, _text, _brackets in parse_edited_txt(
            plan.txt_path.read_text(encoding="utf-8")
        )
    ]

    review_data = build_review_data(
        plan,
        data,
        set(kept_order),
        source_audio=_relative_media(plan.source_wav, out_dir),
        review_audio=_relative_media(wav_path, out_dir),
        display_order=compute_display_order(kept_order, data),
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
