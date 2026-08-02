"""Hlyst — `sc review --serve` の対話モード（削除ON/OFF→試聴を取得→保存）。

127.0.0.1 のローカルサーバー（stdlib のみ）を起こし、レビュー画面から
チェックボックスで削除を切り替え → 試聴音声を取得してキャッシュ（/preview、
自動再生はせずクリック時に再生）→ 保存（/save）できるようにする。
機能は3種類のエンドポイントに限定する（肥大化ガード）:

- `GET /`            : レビュー HTML
- `GET /audio/...`   : 原音・review.wav の2本だけ（allowlist、Range 対応）
- `POST /preview`    : トグル反映済みの計画から窓 wav を返す
- `POST /save`       : .sc.txt へ局所パッチして review.wav を作り直す
- `POST /export`     : （必要なら保存してから）Logic 用 FCPXML を書き出す

**TXT は再構成しない**。起動時の raw 行をそのまま保持し、トグル対象の行の
削除/復活と 〔〕 の付与/解除だけを局所適用する（手編集した句読点・単語削除・
並べ替えを壊さないため）。parse_edited_txt からの逆生成は原文書式を復元できない
ので使わない。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import sys
import threading
import webbrowser
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from . import fillers as fillers_mod
from .fcpxml import export_fcpxml
from .render import (
    _FILLER_RE,
    _LINE_RE,
    BLOCK_SEP,
    FADE_S,
    FILLER_CLOSE,
    FILLER_OPEN,
    GAP_MAX_S,
    GAP_THRESHOLD_S,
    LOW_CONF_MARK,
    build_edit_plan_from_text,
    load_plan_context,
    parse_edited_txt,
    plan_chunk_bounds,
    render_plan,
    render_plan_window,
)
from .review import (
    REVIEW_WAV_SUFFIX,
    build_review_data,
    chunk_output_starts,
    map_source_sample,
    render_html,
)

# バインド先はループバック固定（外部からは到達させない）
HOST = "127.0.0.1"

# POST ボディのサイズ上限。トグル状態は1行あたり数十バイトなので長尺でも十分収まる。
MAX_POST_BYTES = 4 * 1024 * 1024

# 試聴窓の前後余白（秒）。画面の3段目付近で切替できる（既定 ±4秒）。
PREVIEW_PAD_OPTIONS = (2.0, 4.0, 8.0)
PREVIEW_PAD_DEFAULT_S = 4.0

# 配信を許す音声（allowlist）。パスからファイルを組み立てないので traversal 不可。
AUDIO_ROUTES = ("/audio/source", "/audio/review")

# ファイル送出のチャンクサイズ
_SEND_CHUNK = 64 * 1024

# 手編集で一意に対応づけられないときの案内文（保存もプレビューも行わない）
_MANUAL_HINT = "（.sc.txt を直接編集してください）"

# マーカー類（本文の構造照合ではこれらを除いて突き合わせる）
_MARKERS = (LOW_CONF_MARK, BLOCK_SEP, FILLER_OPEN, FILLER_CLOSE)


class TxtEditError(Exception):
    """raw TXT へ局所適用できない（構造照合が一意でない）場合のエラー。"""


class ExternalChangeError(Exception):
    """起動後に .sc.txt が外部で変更されていた場合のエラー。"""


class AudioRenderError(Exception):
    """.sc.txt への書き込みは成功したが review.wav の再生成に失敗した場合のエラー。

    TXT は既にディスクへ保存済みなので、呼び出し側は「保存失敗」ではなく
    「TXT は保存済み・音声の更新のみ失敗」と分かる文言/フラグで応答する。
    """


class FcpxmlExportError(Exception):
    """FCPXML の生成に失敗した場合のエラー（clean export でも起きる）。

    HTTP handler がレスポンスを組み立てられるよう、今回 TXT を書いたか・
    今回 review.wav を再生成したか・確定済み review データを保持する
    （review_updated が無いと stale 再生成成功→FCPXML 失敗のレスポンスで
    音源更新済みかを判定できない）。
    """

    def __init__(
        self, message: str, *,
        txt_saved: bool, review_updated: bool, review_data: dict,
    ) -> None:
        super().__init__(message)
        self.txt_saved = txt_saved
        self.review_updated = review_updated
        self.review_data = review_data


class ExportRefreshError(Exception):
    """/export 中の review.wav 再生成失敗（export 専用）。

    /save の AudioRenderError レスポンス契約（既存 {"error": ..., "txt_saved": true}）
    は一切変更しない。export() は AudioRenderError を捕捉してこの例外へ変換し、
    「今回実際に TXT を書いたか」を保持する（stale 再試行では txt_saved=False に
    なる。true 固定にしない）。
    """

    def __init__(self, message: str, *, txt_saved: bool) -> None:
        super().__init__(message)
        self.txt_saved = txt_saved


# ---------------------------------------------------------------- raw TXT 操作

def detect_newline(text: str) -> str:
    """テキスト中で優勢な改行コードを返す（既定 "\\n"）。"""
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text and "\n" not in text:
        return "\r"
    return "\n"


def split_eol(line: str) -> tuple[str, str]:
    """行を (改行を除いた本体, 改行) に分ける。"""
    for eol in ("\r\n", "\n", "\r"):
        if line.endswith(eol):
            return line[: -len(eol)], eol
    return line, ""


def line_segment_id(line: str) -> str | None:
    """raw 行の ID を返す（`[ID ...] 本文` 形式でなければ None）。"""
    body, _eol = split_eol(line)
    m = _LINE_RE.match(body)
    return m.group(1) if m else None


def line_body_span(line: str) -> tuple[int, int] | None:
    """raw 行の「本文部分」の文字範囲を返す（`[ID 0:00] ` の後ろ〜改行の前）。"""
    body, eol = split_eol(line)
    m = _LINE_RE.match(body)
    if not m:
        return None
    return m.start(2), len(line) - len(eol)


def line_body(line: str) -> str:
    """raw 行の本文部分（`[ID 0:00] ` の後ろ〜改行の前）。ID 行でなければ空文字。"""
    span = line_body_span(line)
    return line[span[0]: span[1]] if span else ""


def uts_label(seg: dict) -> str:
    """UTS 行の標準ラベル（〔〕付き）。transcribe の既定出力と同文言。"""
    duration = max(0.0, float(seg["source_end"]) - float(seg["source_start"]))
    return f"{FILLER_OPEN}未転写発話 {duration:.1f}秒{FILLER_CLOSE}"


def _unrecognized_label(seg: dict) -> str:
    """未認識/UTS 行の標準ラベル（〔〕付き）。transcribe の既定出力と同文言。

    reason:"uts" は「未転写発話」、plain 未認識は「未認識」を使う（どちらも
    auto_uts の既定である〔〕付き＝削除指定の表示）。
    """
    duration = max(0.0, float(seg["source_end"]) - float(seg["source_start"]))
    if seg.get("reason") == "uts":
        return f"{FILLER_OPEN}未転写発話 {duration:.1f}秒{FILLER_CLOSE}"
    return f"{FILLER_OPEN}未認識 {duration:.1f}秒{FILLER_CLOSE}"


def standard_line(seg: dict, fillers: set[str]) -> str:
    """json から transcribe と同じ標準形式の行を生成する（改行なし）。

    起動前から削除済みの行は元の手編集文を知り得ないため、復活時はこの標準形式で
    再生成する（仕様）。
    """
    from .transcribe import _format_time, build_segment_line

    if seg.get("kind") == "unrecognized":
        start = float(seg["source_start"])
        label = _unrecognized_label(seg)
        return f"[{seg['id']} {_format_time(start)}] {label}"

    words = seg.get("words", []) or []
    blocks = (
        [w["block"] for w in words]
        if words and all("block" in w for w in words)
        else None
    )
    suggest = {i for i, w in enumerate(words) if w.get("suggest")}
    line, _count = build_segment_line(
        seg["id"], words, fillers, True, blocks=blocks, suggest_indices=suggest
    )
    return line


def _clean_body_with_map(body: str) -> tuple[str, list[int]]:
    """本文からマーカー（◆／〔〕）を除いた文字列と、各文字の元 index を返す。"""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(body):
        if ch in _MARKERS:
            continue
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def locate_word_spans(body: str, words: list[dict]) -> dict[int, tuple[int, int]]:
    """本文中で各単語の本体文字が占める範囲を {word_index: (start, end)} で返す。

    json の word_index は手編集後 TXT の文字位置と対応しない（同一フィラーが複数
    出現すると単純な文字検索は誤爆する）ため、json 単語列の連結とマーカー
    （◆ ／ 〔〕）を除いた本文とを文字 diff で突き合わせ、`equal`（＝同一と確定
    した文字）だけを対応づける。本体文字が全て対応し raw 上で連続している単語
    だけを返す（対応が一意でない単語は結果に現れない）。
    """
    cleaned, index_map = _clean_body_with_map(body)
    offsets: list[tuple[int, int]] = []
    parts: list[str] = []
    pos = 0
    for w in words:
        text = w.get("word", "").replace(BLOCK_SEP, "")
        offsets.append((pos, pos + len(text)))
        parts.append(text)
        pos += len(text)
    original = "".join(parts)

    mapping: dict[int, int] = {}
    sm = SequenceMatcher(None, original, cleaned, autojunk=False)
    for tag, i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                mapping[i1 + k] = j1 + k

    out: dict[int, tuple[int, int]] = {}
    for word_index, (p0, p1) in enumerate(offsets):
        # 単語先頭の空白（英語トークンの区切り）は〔〕の外側に置くので本体から除く
        core = [p for p in range(p0, p1) if not original[p].isspace()]
        if not core:
            continue
        raws: list[int] = []
        for p in core:
            c = mapping.get(p)
            if c is None:
                break
            raws.append(index_map[c])
        if len(raws) != len(core):
            continue
        if raws != list(range(raws[0], raws[0] + len(raws))):
            continue
        out[word_index] = (raws[0], raws[-1] + 1)
    return out


def locate_word_span(
    body: str, words: list[dict], word_index: int
) -> tuple[int, int] | None:
    """本文中で words[word_index] の本体文字が占める範囲（無ければ None）。"""
    return locate_word_spans(body, words).get(word_index)


def bracket_spans(body: str) -> list[tuple[int, int]]:
    """本文中の 〔…〕 の範囲列。"""
    return [(m.start(), m.end()) for m in _FILLER_RE.finditer(body)]


def bracket_word_spans(body: str, words: list[dict]) -> dict[int, tuple[int, int]]:
    """〔〕で囲まれている単語を {word_index: 〔〕の範囲} で返す。

    対応づけは「その単語の本体文字が〔〕の内側にあるか」という位置関係で行う。
    同一フィラーが複数出現し片方だけ〔〕が付いている場合でも、付いている方の
    出現を特定できる（内容の順序マッチでは先頭の候補を拾ってしまう）。
    """
    located = locate_word_spans(body, words)
    out: dict[int, tuple[int, int]] = {}
    for start, end in bracket_spans(body):
        inner = (start + len(FILLER_OPEN), end - len(FILLER_CLOSE))
        for word_index, (s, e) in located.items():
            if inner[0] <= s and e <= inner[1]:
                out[word_index] = (start, end)
    return out


def word_requested_delete(body: str, words: list[dict], word_index: int) -> bool:
    """単語が TXT 上で削除指定されているか（〔〕あり、または本文から消えている）。"""
    if word_index in bracket_word_spans(body, words):
        return True
    return locate_word_span(body, words, word_index) is None


def _ambiguous_duplicate_word(body: str, words: list[dict], word_index: int) -> bool:
    """word_index と同一の本体テキストを持つ語が同じセグメント内に複数あり、

    かつそのうち1語でも raw へ一意対応できていない（locate_word_spans に
    現れない）場合に True を返す。SequenceMatcher の整列は同一テキストが
    複数出現すると恣意的になるため、そのグループのどれか1語でも対応が
    崩れていれば、対象語自身が一意に見えても保守的に不確定とみなす。
    """
    if not (0 <= word_index < len(words)):
        return False
    text = words[word_index].get("word", "")
    siblings = [i for i, w in enumerate(words) if w.get("word", "") == text]
    if len(siblings) < 2:
        return False
    located = locate_word_spans(body, words)
    return any(i not in located for i in siblings)


def set_word_bracket(
    body: str, words: list[dict], word_index: int, delete: bool
) -> str:
    """本文の単語 word_index に 〔〕 を付与/解除した本文を返す。

    既に目的の状態なら本文をそのまま返す（未変更行をバイト単位で不変にするため）。
    構造照合で raw 上の文字範囲を一意に特定できない（手動で消された等）場合は
    TxtEditError を送出し、呼び出し側で保存せずに手動編集を案内する。

    同一テキストの語が同じセグメント内に複数あり、そのうち1語でも一意対応
    できていない場合も、対象語自身が特定できて見えるかにかかわらず保守的に
    TxtEditError にする（SequenceMatcher の整列は同一語の重複に対して恣意的
    なため、見かけ上の対応を信用しない）。
    """
    text = words[word_index].get("word", "").strip() if word_index < len(words) else ""
    if _ambiguous_duplicate_word(body, words, word_index):
        raise TxtEditError(
            f"「{text}」と同じ文言の語が複数あり、raw 上で一意に対応づけ"
            f"られません{_MANUAL_HINT}"
        )
    spans = bracket_word_spans(body, words)
    if delete:
        if word_index in spans:
            return body
        located = locate_word_span(body, words, word_index)
        if located is None:
            raise TxtEditError(
                f"「{text}」の位置を手編集後の行から特定できませんでした{_MANUAL_HINT}"
            )
        s, e = located
        return body[:s] + FILLER_OPEN + body[s:e] + FILLER_CLOSE + body[e:]

    if word_index not in spans:
        if locate_word_span(body, words, word_index) is None:
            raise TxtEditError(
                f"「{text}」は手動で削除されているため画面からは復活できません{_MANUAL_HINT}"
            )
        return body
    s, e = spans[word_index]
    inner = body[s + len(FILLER_OPEN): e - len(FILLER_CLOSE)]
    return body[:s] + inner + body[e:]


def set_line_bracket(body: str, seg: dict, delete: bool) -> str:
    """未認識/UTS 行の本文に 〔〕 を付与/解除する。

    未認識/UTS 行の本文は機械生成のプレースホルダなので、付与時は標準ラベルへ
    揃える。解除は 〔〕 の記号だけを外して中身を残す（同じ操作で元に戻る）。
    """
    has_bracket = _FILLER_RE.search(body) is not None
    if delete == has_bracket:
        return body
    if delete:
        return _unrecognized_label(seg)
    return _FILLER_RE.sub(
        lambda m: m.group(0)[len(FILLER_OPEN): -len(FILLER_CLOSE)], body
    )


def _is_bracket_toggle(seg: dict) -> bool:
    """この行の「削除」を 〔〕 で表すか（未認識/UTS 行）／行の有無で表すか。

    plain 未認識行（reason なし）も UTS と同じく auto_uts の既定で〔〕付き
    出力になったため、UTS 限定だった bracket toggle をここで揃える。
    """
    return seg.get("kind") == "unrecognized"


def _line_index(lines: list[str]) -> dict[str, int]:
    return {
        seg_id: i
        for i, line in enumerate(lines)
        if (seg_id := line_segment_id(line)) is not None
    }


def _insert_position(
    lines: list[str], order: list[str], seg_id: str
) -> int:
    """復活行の挿入位置（json 順で直後の生存行の前、無ければ直前の生存行の後ろ）。"""
    index = _line_index(lines)
    pos = order.index(seg_id)
    for other in order[pos + 1:]:
        if other in index:
            return index[other]
    for other in reversed(order[:pos]):
        if other in index:
            return index[other] + 1
    return len(lines)


def _replace_body(line: str, new_body: str) -> str:
    span = line_body_span(line)
    if span is None:
        return line
    s, e = span
    return line[:s] + new_body + line[e:]


def apply_toggles(
    raw_text: str, data: dict, fillers: set[str], toggles: list[dict]
) -> str:
    """raw TXT にトグル状態を局所適用した新しいテキストを返す。

    現在の状態と一致するトグルは何もしない（未変更行はバイト単位で不変）。
    行の削除/復活を先に処理してから、単語（〔〕）のトグルを適用する。
    """
    segments = {seg["id"]: seg for seg in data.get("segments", [])}
    order = [seg["id"] for seg in data.get("segments", [])]
    lines = raw_text.splitlines(keepends=True)
    eol = detect_newline(raw_text)

    line_toggles = [t for t in toggles if t.get("type", "line") == "line"]
    word_toggles = [t for t in toggles if t.get("type") == "word"]
    rank = {seg_id: i for i, seg_id in enumerate(order)}
    line_toggles.sort(key=lambda t: rank.get(t["id"], 0))

    for toggle in line_toggles:
        seg = segments[toggle["id"]]
        delete = bool(toggle.get("delete"))
        index = _line_index(lines)
        present = toggle["id"] in index

        if _is_bracket_toggle(seg):
            if not present:
                if delete:
                    continue  # 行ごと消えている＝削除指定のまま。何もしない
                # 行を標準形式で復活させたうえで〔〕を外す（＝音声保持）
                new_line = standard_line(seg, fillers) + eol
                new_line = _replace_body(
                    new_line, set_line_bracket(line_body(new_line), seg, False)
                )
                lines = _insert_line(lines, order, toggle["id"], new_line, eol)
                continue
            i = index[toggle["id"]]
            lines[i] = _replace_body(
                lines[i], set_line_bracket(line_body(lines[i]), seg, delete)
            )
            continue

        if delete and present:
            del lines[index[toggle["id"]]]
        elif not delete and not present:
            lines = _insert_line(
                lines, order, toggle["id"], standard_line(seg, fillers) + eol, eol
            )

    for toggle in word_toggles:
        seg = segments[toggle["id"]]
        index = _line_index(lines)
        if toggle["id"] not in index:
            continue  # 行ごと削除されている行の単語トグルは無効
        words = seg.get("words", []) or []
        word_index = int(toggle.get("word", -1))
        if not (0 <= word_index < len(words)):
            continue
        i = index[toggle["id"]]
        new_body = set_word_bracket(
            line_body(lines[i]), words, word_index, bool(toggle.get("delete")),
        )
        lines[i] = _replace_body(lines[i], new_body)

    return "".join(lines)


def _insert_line(
    lines: list[str], order: list[str], seg_id: str, new_line: str, eol: str
) -> list[str]:
    """復活行を json 順の位置へ挿入する（末尾追加時は前行の改行を補う）。"""
    pos = _insert_position(lines, order, seg_id)
    lines = list(lines)
    if pos == len(lines) and lines:
        last, last_eol = split_eol(lines[-1])
        if not last_eol:
            lines[-1] = last + eol
    lines.insert(pos, new_line)
    return lines


def requested_state(raw_text: str, data: dict, fillers: set[str]) -> dict:
    """raw TXT から「削除指定（requested_delete）」の一覧を組み立てる。

    戻り値 {"segments": {id: bool}, "words": {id: {word_index: bool}}}。
    未認識/UTS 行は 〔〕 の有無、その他の行は行の有無が削除指定にあたる。
    """
    lines = raw_text.splitlines(keepends=True)
    index = _line_index(lines)
    seg_out: dict[str, bool] = {}
    word_out: dict[str, dict[int, bool]] = {}
    for seg in data.get("segments", []):
        seg_id = seg["id"]
        present = seg_id in index
        body = line_body(lines[index[seg_id]]) if present else ""
        if _is_bracket_toggle(seg):
            seg_out[seg_id] = (not present) or bool(_FILLER_RE.search(body))
        else:
            seg_out[seg_id] = not present
        words = seg.get("words", []) or []
        if not words:
            continue
        states: dict[int, bool] = {}
        for word_index in range(len(words)):
            states[word_index] = (
                True if not present
                else word_requested_delete(body, words, word_index)
            )
        word_out[seg_id] = states
    return {"segments": seg_out, "words": word_out}


# ---------------------------------------------------------------- セッション

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_wav(samples, samplerate: int, subtype: str | None) -> bytes:
    """numpy 配列を wav バイト列（メモリ上）にする。"""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, samplerate, format="WAV", subtype=subtype)
    return buf.getvalue()


class HlystSession:
    """1つの .sc.txt に対する対話セッション（計画・試聴・保存を直列化して持つ）。

    preview / save と保存基準（raw TXT・mtime・ハッシュ）の更新は同じロックで
    直列化する（同時実行で基準がずれると2回目の保存が外部変更扱いになるため）。
    """

    def __init__(
        self,
        txt_path: str | Path,
        gap_threshold: float = GAP_THRESHOLD_S,
        gap_max: float = GAP_MAX_S,
    ) -> None:
        self.context = load_plan_context(txt_path)
        self.gap_threshold = gap_threshold
        self.gap_max = gap_max
        self.fillers = fillers_mod.load_fillers(self.context.data.get("language"))
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.raw_text = self.context.txt_path.read_text(encoding="utf-8")
        self.digest = _sha256(self.raw_text)
        self.mtime_ns = self.context.txt_path.stat().st_mtime_ns
        self.review_wav = (
            self.context.txt_path.parent
            / f"{self.context.base_name}{REVIEW_WAV_SUFFIX}"
        )
        self.version = 0
        self.plan = None
        # review.wav が現在の raw TXT より古いか（再生成が必要か）。
        # False へ戻すのは render_plan 成功後だけ（refresh に責務を集約）。
        self.review_stale = True
        self.refresh()

    # -- 計画とレビュー音声

    def plan_for(self, text: str):
        return build_edit_plan_from_text(
            self.context, text,
            gap_threshold=self.gap_threshold, gap_max=self.gap_max,
        )

    def refresh(self) -> None:
        """現在の raw TXT から計画を作り直し review.wav を書き出す。

        review_stale を False へ戻すのは render_plan が成功した後だけ。
        計画を作り直した時点で戻すと、書き出しに失敗したまま「最新」とみなされ
        次回以降に再試行されなくなる。
        """
        self.plan = self.plan_for(self.raw_text)
        render_plan(self.plan, self.review_wav)
        self.review_stale = False
        self.version += 1

    def audio_url(self, name: str) -> str:
        return f"/audio/{name}?token={quote(self.token)}"

    def review_data(self, text: str, plan) -> dict:
        kept_ids = {seg_id for seg_id, _t, _b in parse_edited_txt(text)}
        return build_review_data(
            plan,
            self.context.data,
            kept_ids,
            source_audio=self.audio_url("source"),
            review_audio=self.audio_url("review"),
            serve=True,
            requested=requested_state(text, self.context.data, self.fillers),
            token=self.token,
        )

    def html(self) -> str:
        with self.lock:
            return render_html(self.review_data(self.raw_text, self.plan))

    def audio_path(self, route: str) -> Path:
        return (
            self.context.source_wav if route == "/audio/source" else self.review_wav
        )

    # -- 試聴

    def preview(
        self, toggles: list[dict], target: dict, pad_s: float = PREVIEW_PAD_DEFAULT_S
    ) -> dict:
        """トグル反映済みの計画から対象周辺の窓 wav を作って返す。

        pad_s は試聴窓の前後余白（秒）。呼び出し側（HTTP ハンドラ）が
        PREVIEW_PAD_OPTIONS で検証済みの値を渡す前提。
        """
        with self.lock:
            text = apply_toggles(
                self.raw_text, self.context.data, self.fillers, toggles
            )
            plan = self.plan_for(text)
            data = self.review_data(text, plan)
            info = preview_center(plan, data, target)
            center = info["center"]
            window, start_sample = render_plan_window(plan, center, pad_s)
            wav = encode_wav(
                window, plan.samplerate, plan.audio_info.subtype
            )
            samplerate = plan.samplerate
            window_start_s = (start_sample / samplerate) if samplerate else 0.0
            window_end_s = (
                window_start_s + (len(window) / samplerate) if samplerate else 0.0
            )
            return {
                "audio": base64.b64encode(wav).decode("ascii"),
                "center": center,
                "center_s": center,
                "target_source_s": info["target_source_s"],
                "window_start_s": window_start_s,
                "window_end_s": window_end_s,
                "pad_s": pad_s,
                "message": info["message"],
                "segments": data["segments"],
                "stats": data["stats"],
            }

    # -- 保存（save / export が共用する部品。ロックは呼び出し側で取る）

    def _check_external_change(self) -> None:
        """起動後に .sc.txt が外部で変更されていたら ExternalChangeError。"""
        # 判定は hash 不一致だけを基準にする（mtime は偽装され得るため
        # 「mtime 一致＝無検査で信頼してよい」とはしない）。mtime 一致なら
        # hash 計算を省略してよいという高速化はここでは採用しない。
        current = self.context.txt_path.read_text(encoding="utf-8")
        if _sha256(current) != self.digest:
            raise ExternalChangeError(
                "起動後に .sc.txt が外部で変更されています。"
                "画面を閉じて `sc review --serve` をやり直してください"
            )

    def _apply_toggles_text(self, toggles: list[dict]) -> str:
        """トグルを raw TXT へ局所適用した新しいテキストを返す。"""
        return apply_toggles(
            self.raw_text, self.context.data, self.fillers, toggles
        )

    def _write_if_changed(self, text: str) -> bool:
        """内容が変わっていれば .sc.txt を書き換える（書いたら True）。"""
        path = self.context.txt_path
        if text == self.raw_text:
            return False
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():
            backup.write_text(self.raw_text, encoding="utf-8")
        _atomic_write(path, text)
        # 保存基準を更新する（さもないと2回目の保存が外部変更扱いになる）
        self.raw_text = text
        self.digest = _sha256(text)
        self.mtime_ns = path.stat().st_mtime_ns
        # 書いた時点で review.wav は古い。再生成成功まで True のままにする。
        self.review_stale = True
        return True

    def _refresh_after_save(self) -> None:
        """TXT 書き込み後の review.wav 再生成（失敗は AudioRenderError）。

        呼ばれる時点で TXT の書き込みは既に完了しているので、失敗しても
        「保存失敗」ではなく「TXT は保存済み・音声の更新のみ失敗」と
        呼び出し側に伝える（TxtEditError/ExternalChangeError と違い、
        ロールバックすべきディスク上の状態が既に無いため）。
        """
        try:
            self.refresh()
        except Exception as exc:
            raise AudioRenderError(
                "TXT は保存済みです。review.wav の更新に失敗しました: "
                f"{exc}"
            ) from exc

    def save(self, toggles: list[dict]) -> dict:
        """トグル状態を .sc.txt へ局所パッチし、review.wav を作り直す。"""
        with self.lock:
            self._check_external_change()
            self._write_if_changed(self._apply_toggles_text(toggles))
            self._refresh_after_save()
            data = self.review_data(self.raw_text, self.plan)
            return {
                "version": self.version,
                "txt_path": str(self.context.txt_path),
                "segments": data["segments"],
                "stats": data["stats"],
                "warnings": data["warnings"],
            }

    # -- FCPXML 書き出し

    @property
    def fcpxml_path(self) -> Path:
        """書き出し先（sc logic の既定出力名と同一）。"""
        return (
            self.context.txt_path.parent
            / f"{self.context.base_name}.logic.fcpxml"
        )

    def export(self, toggles: list[dict]) -> dict:
        """未保存変更があれば保存してから Logic 用 FCPXML を書き出す。

        変更が無く review.wav も最新なら TXT・review.wav・version を触らず
        FCPXML だけを作り直す。前回の export で review.wav の再生成に失敗して
        いた場合（review_stale）は、TXT を書き直さずに再生成だけ再試行する。
        """
        with self.lock:
            self._check_external_change()
            saved = self._write_if_changed(self._apply_toggles_text(toggles))

            review_updated = False
            if saved or self.review_stale:
                try:
                    self._refresh_after_save()
                except AudioRenderError as exc:
                    # review.wav が古いままなので FCPXML は作らない。
                    # txt_saved は「今回実際に書いたか」（stale 再試行では False）。
                    # _refresh_after_save の文言は「TXT は保存済みです。…」で
                    # 始まる固定文言だが、stale 再試行（saved=False）ではその
                    # 前置きが誤解を招くため、export() 側で txt_saved に応じて
                    # 文言を組み立てる。
                    if saved:
                        message = str(exc)
                    else:
                        cause = exc.__cause__ if exc.__cause__ is not None else exc
                        message = f"review.wavの更新に失敗しました: {cause}"
                    raise ExportRefreshError(message, txt_saved=saved) from exc
                review_updated = True

            data = self.review_data(self.raw_text, self.plan)
            final_path = self.fcpxml_path
            stats_lines: list[str] = []  # 標準出力抑止専用。レスポンスへは返さない
            tmp = final_path.with_name(f".{final_path.name}.hlyst.tmp")
            try:
                export_fcpxml(
                    str(self.context.txt_path),
                    output=str(tmp),
                    gap_threshold=self.gap_threshold,
                    gap_max=self.gap_max,
                    stats_out=stats_lines,
                )
                os.replace(tmp, final_path)
            except Exception as exc:
                try:
                    tmp.unlink(missing_ok=True)  # 一時ファイルを残さない
                except OSError:
                    pass  # cleanup 失敗で元例外を隠さない
                raise FcpxmlExportError(
                    f"FCPXMLの生成に失敗しました: {exc}",
                    txt_saved=saved,
                    review_updated=review_updated,
                    review_data=data,
                ) from exc  # 既存の正常な .logic.fcpxml は不変

            return {
                "fcpxml_path": str(final_path),
                "fcpxml_generated": True,
                "txt_saved": saved,
                "review_updated": review_updated,
                "version": self.version,
                "segments": data["segments"],
                "stats": data["stats"],
                "warnings": data["warnings"],
            }


def _atomic_write(path: Path, text: str) -> None:
    """同ディレクトリの一時ファイルへ書いてから rename する（原子的置換）。

    元ファイルが存在する場合はそのパーミッションを一時ファイルへ引き継ぐ
    （さもないと umask 依存の既定パーミッションで置き換わってしまう）。
    """
    tmp = path.with_name(f".{path.name}.hlyst.tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        os.chmod(tmp, os.stat(path).st_mode)
    os.replace(tmp, path)


def preview_center(plan, data: dict, target: dict) -> dict:
    """試聴の中心時刻（新プランの出力時刻）・対象の原音時刻・通知文言を返す。

    削除OFF（復活試聴）は対象中央を新プランの写像で出力時刻へ変換する。
    削除ON（削除試聴）は対象が新出力に存在しないため、削除後の前後区間の
    結合境界を中心にする（単純な source→output 変換では扱えない）。

    戻り値 {"center": 出力時刻(秒), "target_source_s": 対象の原音時刻(秒),
    "message": 通知文言}。
    """
    samplerate = plan.samplerate
    bounds, flags, _ = plan_chunk_bounds(plan)
    starts, _total = chunk_output_starts(
        bounds, flags, int(round(FADE_S * samplerate))
    )
    seg = next((s for s in data["segments"] if s["id"] == target.get("id")), None)
    if seg is None:
        raise ValueError(f"json に存在しない ID です: [{target.get('id')}]")

    def _sec(sample: int | None) -> float:
        return (sample / samplerate) if (sample is not None and samplerate) else 0.0

    if target.get("type") == "word":
        word_index = int(target.get("word", -1))
        word = next((w for w in seg["words"] if w["index"] == word_index), None)
        if word is None:
            raise ValueError(f"単語が見つかりません: [{seg['id']}] {word_index}")
        target_source_s = float(word["center"])
        mid = int(round(target_source_s * samplerate))
        center = _sec(map_source_sample(bounds, starts, mid))
        requested = word["requested_delete"]
        effective = word["effective_delete"]
    else:
        target_source_s = (float(seg["source"]) + float(seg["source_end"])) / 2.0
        if seg["effective_delete"]:
            # 削除済み → 結合境界（build_review_data が算出済み）
            center = seg["output"]
        else:
            mid = int(round(target_source_s * samplerate))
            center = _sec(map_source_sample(bounds, starts, mid))
        requested = seg["requested_delete"]
        effective = seg["effective_delete"]

    if requested and not effective:
        reason = plan.guarded_cut_reasons.get(seg["id"])
        message = (
            "現在案プレビュー：ポーズ不足のため保持"
            if reason == "pause"
            else "現在案プレビュー：安全判定により保持"
        )
    elif requested:
        message = "現在案プレビュー：削除"
    else:
        message = "現在案プレビュー：保持"
    return {"center": center, "target_source_s": target_source_s, "message": message}


# ---------------------------------------------------------------- HTTP

def parse_byte_range(header: str | None, size: int):
    """Range ヘッダを (start, end)（両端含む）へ解釈する（純関数）。

    戻り値: None＝範囲指定なし/非対応形式（全体を返す）、"invalid"＝416、
    (start, end)＝206。複数レンジは非対応なので全体を返す。
    """
    if not header:
        return None
    spec = header.strip()
    if not spec.lower().startswith("bytes="):
        return None
    spec = spec[len("bytes="):].strip()
    if "," in spec:
        return None
    if "-" not in spec:
        return "invalid"
    first, _sep, last = spec.partition("-")
    first, last = first.strip(), last.strip()
    try:
        if first == "":
            if last == "":
                return "invalid"
            suffix = int(last)
            if suffix <= 0:
                return "invalid"
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
    except ValueError:
        return "invalid"
    if size <= 0 or start < 0 or start >= size or start > end:
        return "invalid"
    return start, min(end, size - 1)


class _Handler(BaseHTTPRequestHandler):
    """Hlyst の最小 HTTP ハンドラ。

    /save がローカルファイルを書くため、127.0.0.1 バインドに加えて
    起動毎トークン（URL＋POST ヘッダ）と Host/Origin の完全一致検証を行う。
    """

    server_version = "Hlyst"
    protocol_version = "HTTP/1.1"

    # -- 送出ヘルパ

    def log_message(self, fmt, *args):  # noqa: D102 - アクセスログは出さない
        pass

    def _common_headers(self) -> None:
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _send_bytes(self, status: int, ctype: str, body: bytes,
                    close: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, obj: dict, close: bool = False) -> None:
        self._send_bytes(
            status, "application/json; charset=utf-8",
            json.dumps(obj, ensure_ascii=False).encode("utf-8"),
            close=close,
        )

    def _error(self, status: int, message: str) -> None:
        # 拒否時はボディを読み切っていないことがあるので keep-alive を切る
        self._send_json(status, {"error": message}, close=True)

    # -- 検証

    @property
    def session(self) -> HlystSession:
        return self.server.session  # type: ignore[attr-defined]

    def _origin(self) -> str:
        return f"http://{HOST}:{self.server.server_address[1]}"

    def _check_headers(self, require_origin: bool) -> bool:
        """Host / Origin の完全一致検証（DNS rebinding・CSRF 対策）。"""
        expected_host = f"{HOST}:{self.server.server_address[1]}"
        if (self.headers.get("Host") or "").strip() != expected_host:
            self._error(403, "Host ヘッダが不正です")
            return False
        origin = (self.headers.get("Origin") or "").strip()
        if origin:
            if origin != self._origin():
                self._error(403, "Origin ヘッダが不正です")
                return False
        elif require_origin:
            self._error(403, "Origin ヘッダがありません")
            return False
        return True

    def _check_token(self, query: dict, header: bool) -> bool:
        token = (query.get("token") or [""])[0]
        if not secrets.compare_digest(token, self.session.token):
            self._error(403, "トークンが不正です")
            return False
        if header:
            sent = (self.headers.get("X-Hlyst-Token") or "").strip()
            if not secrets.compare_digest(sent, self.session.token):
                self._error(403, "トークンヘッダが不正です")
                return False
        return True

    # -- GET

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler の規約
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._check_headers(require_origin=False):
            return
        if not self._check_token(query, header=False):
            return

        if parsed.path == "/":
            body = self.session.html().encode("utf-8")
            self._send_bytes(200, "text/html; charset=utf-8", body)
            return
        if parsed.path in AUDIO_ROUTES:
            self._send_audio(self.session.audio_path(parsed.path))
            return
        self._error(404, "見つかりません")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _send_audio(self, path: Path) -> None:
        if not path.exists():
            self._error(404, "音声がありません")
            return
        size = path.stat().st_size
        rng = parse_byte_range(self.headers.get("Range"), size)
        if rng == "invalid":
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.send_header("Accept-Ranges", "bytes")
            self._common_headers()
            self.end_headers()
            return

        if rng is None:
            start, end = 0, size - 1
            status = 200
        else:
            start, end = rng
            status = 206
        length = end - start + 1 if size else 0

        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._common_headers()
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                block = fh.read(min(_SEND_CHUNK, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    # -- POST

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._check_headers(require_origin=True):
            return
        if not self._check_token(query, header=True):
            return
        if parsed.path not in ("/preview", "/save", "/export"):
            self._error(404, "見つかりません")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "Content-Length が不正です")
            return
        if length <= 0:
            self._error(400, "ボディがありません")
            return
        if length > MAX_POST_BYTES:
            self._error(413, "ボディが大きすぎます")
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._error(400, "JSON を解釈できません")
            return
        if not isinstance(payload, dict):
            self._error(400, "JSON オブジェクトが必要です")
            return

        try:
            toggles = _validate_toggles(
                payload.get("toggles"), self.session.context.data
            )
        except ValueError as exc:
            self._error(400, str(exc))
            return

        try:
            if parsed.path == "/preview":
                target = payload.get("target")
                if not isinstance(target, dict) or not isinstance(
                    target.get("id"), str
                ):
                    self._error(400, "target が不正です")
                    return
                try:
                    pad_s = _validate_pad_s(payload.get("pad_s"))
                except ValueError as exc:
                    self._error(400, str(exc))
                    return
                self._send_json(200, self.session.preview(toggles, target, pad_s))
            elif parsed.path == "/save":
                self._send_json(200, self.session.save(toggles))
            else:
                self._send_json(200, self.session.export(toggles))
        except ExportRefreshError as exc:
            # review.wav が古いままなので plan 由来のデータは返さない。
            # txt_saved は「今回実際に TXT を書いたか」。
            self._send_json(
                500,
                {
                    "error": str(exc),
                    "txt_saved": exc.txt_saved,
                    "fcpxml_generated": False,
                    "review_updated": False,
                },
                close=True,
            )
        except FcpxmlExportError as exc:
            # TXT 保存・review.wav 再生成の成否は 2軸で伝える（画面は
            # txt_saved で保存基準、review_updated で音源を同期する）。
            data = exc.review_data
            self._send_json(
                500,
                {
                    "error": str(exc),
                    "txt_saved": exc.txt_saved,
                    "fcpxml_generated": False,
                    "review_updated": exc.review_updated,
                    "version": self.session.version,
                    "segments": data["segments"],
                    "stats": data["stats"],
                    "warnings": data["warnings"],
                },
                close=True,
            )
        except AudioRenderError as exc:
            # TXT の書き込みは成功しているので "保存失敗" とは区別する
            # （txt_saved フラグと文言の両方で伝える）。
            self._send_json(
                500, {"error": str(exc), "txt_saved": True}, close=True
            )
        except ExternalChangeError as exc:
            self._error(409, str(exc))
        except TxtEditError as exc:
            self._error(409, str(exc))
        except ValueError as exc:
            self._error(400, str(exc))
        except OSError as exc:
            self._error(500, f"入出力エラー: {exc}")


def _validate_pad_s(value) -> float:
    """POST の pad_s を検証して返す（未指定は既定値、他は allowlist で拒否）。"""
    if value is None:
        return PREVIEW_PAD_DEFAULT_S
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"pad_s が不正です: {value!r}")
    pad = float(value)
    if pad not in PREVIEW_PAD_OPTIONS:
        raise ValueError(f"pad_s が不正です: {value!r}")
    return pad


def _validate_toggles(toggles, data: dict) -> list[dict]:
    """POST のトグル配列を検証して返す（未知 ID・不正な型は拒否）。"""
    if not isinstance(toggles, list):
        raise ValueError("toggles が配列ではありません")
    known = {seg["id"] for seg in data.get("segments", [])}
    out: list[dict] = []
    for item in toggles:
        if not isinstance(item, dict):
            raise ValueError("toggles の要素が不正です")
        seg_id = item.get("id")
        if not isinstance(seg_id, str) or seg_id not in known:
            raise ValueError(f"json に存在しない ID です: [{seg_id}]")
        kind = item.get("type", "line")
        if kind not in ("line", "word"):
            raise ValueError(f"不明なトグル種別です: {kind}")
        entry = {"type": kind, "id": seg_id, "delete": bool(item.get("delete"))}
        if kind == "word":
            word = item.get("word")
            if not isinstance(word, int) or isinstance(word, bool) or word < 0:
                raise ValueError("word index が不正です")
            entry["word"] = word
        out.append(entry)
    return out


class HlystServer(ThreadingHTTPServer):
    """セッションを保持する ThreadingHTTPServer。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, session: HlystSession, port: int = 0) -> None:
        super().__init__((HOST, port), _Handler)
        self.session = session


def create_server(
    txt_path: str | Path,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
    port: int = 0,
) -> tuple[HlystServer, str]:
    """セッションを作り 127.0.0.1 のサーバーを立てて (server, URL) を返す。"""
    session = HlystSession(txt_path, gap_threshold=gap_threshold, gap_max=gap_max)
    httpd = HlystServer(session, port)
    url = (
        f"http://{HOST}:{httpd.server_address[1]}/"
        f"?token={quote(session.token)}"
    )
    return httpd, url


def serve_review(
    txt_path: str,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
    open_browser: bool = True,
    port: int = 0,
) -> dict:
    """Hlyst 対話モードを起動し、Ctrl-C まで待ち受ける。"""
    httpd, url = create_server(
        txt_path, gap_threshold=gap_threshold, gap_max=gap_max, port=port
    )
    session = httpd.session
    # パイプ/リダイレクト先でも URL がすぐ見えるよう flush する
    print(f"生成: {session.review_wav}", flush=True)
    print(f"Hlyst: {url}", flush=True)
    print("終了するには Ctrl-C を押してください", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
    finally:
        httpd.shutdown()
        httpd.server_close()
    return {"wav_path": str(session.review_wav), "url": url}
