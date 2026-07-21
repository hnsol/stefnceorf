"""音声の再構成 (render)。

編集後の .sc.txt と .sc.json を突き合わせ、残す単語の時間区間を算出し、
元 wav から該当区間を切り出してクロスフェード結合し出力する。

diff→残す単語→時間区間の算出は音声 I/O から分離した純関数として実装している
（テスト容易性のため）。
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# transcribe 側と同じマーカー定義
LOW_CONF_MARK = "◆"
FILLER_OPEN = "〔"
FILLER_CLOSE = "〕"

# 区間前後に付与するマージン（秒）と結合クロスフェード長（秒）
MARGIN_S = 0.02
FADE_S = 0.008

# `[ID] text` 行のパターン。`[...]` 内の最初の空白区切りトークンをIDとし、以降は無視。
# 新形式 `[0001 0:12] text` でも旧形式 `[0001] text` でも動く。
_LINE_RE = re.compile(r"^\[([^\]\s]+)(?:\s+[^\]]*)?\]\s?(.*)$")
# 〔...〕（フィラー提案）を中身ごと削除するためのパターン
_FILLER_RE = re.compile(re.escape(FILLER_OPEN) + r"[^" + re.escape(FILLER_CLOSE) + r"]*" + re.escape(FILLER_CLOSE))


def _clean_edited_text(text: str) -> str:
    """編集後テキストから ◆ を除去し、〔...〕 を中身ごと削除する。

    括弧が外されている（〔〕が無い）箇所は通常の文字として残る。
    """
    text = _FILLER_RE.sub("", text)
    text = text.replace(LOW_CONF_MARK, "")
    return text


def parse_edited_txt(text: str) -> list[tuple[str, str]]:
    """編集後 txt をパースして (ID, クリーン済みテキスト) の列を返す。

    - 空行（空白のみ含む）は無視する
    - `[ID] text` 形式でない非空行は ValueError
    - ◆ は除去、〔...〕 は削除扱いへ変換する
    - ID の重複は ValueError
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        m = _LINE_RE.match(raw)
        if not m:
            raise ValueError(
                f"{lineno}行目: ID の無い行です（`[ID] テキスト` 形式が必要）: {raw!r}"
            )
        seg_id = m.group(1)
        if seg_id in seen:
            raise ValueError(f"{lineno}行目: ID が重複しています: [{seg_id}]")
        seen.add(seg_id)
        out.append((seg_id, _clean_edited_text(m.group(2))))
    return out


def surviving_words(
    original_words: list[str], edited_text: str
) -> tuple[set[int], list[str]]:
    """元単語列と編集後テキストの文字 diff から「残す単語」の集合を求める。

    - equal: 元の文字が生存
    - delete: 元の文字が削除 → その文字を含む単語は削除扱い
    - replace / insert: 追加・書き換えは無視して警告（元文字は生存扱いにする）

    単語は全文字が生存していれば残し、1文字でも削除されていれば削除する。
    戻り値: (残す単語 index の集合, 警告メッセージ列)
    """
    original = "".join(original_words)
    survived = [False] * len(original)
    warnings: list[str] = []

    sm = SequenceMatcher(None, original, edited_text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                survived[k] = True
        elif tag == "replace":
            # 書き換えは無視（元文字は残す扱い）。警告のみ。
            for k in range(i1, i2):
                survived[k] = True
            warnings.append(
                f"書き換えを無視しました: {original[i1:i2]!r} → {edited_text[j1:j2]!r}"
            )
        elif tag == "insert":
            warnings.append(f"追加を無視しました: {edited_text[j1:j2]!r}")
        # delete: 何もしない（生存フラグ False のまま）

    keep: set[int] = set()
    pos = 0
    for wi, w in enumerate(original_words):
        length = len(w)
        # 単語先頭・末尾の空白（英語トークンの整形上の区切り）は diff の
        # 曖昧性で削除側に寄りやすいため、生存判定は非空白文字のみで行う。
        core_positions = [
            pos + k for k, ch in enumerate(w) if not ch.isspace()
        ]
        if not core_positions or all(survived[p] for p in core_positions):
            keep.add(wi)
        pos += length
    return keep, warnings


def words_to_intervals(
    words: list[dict],
    keep_indices: set[int],
    margin: float = MARGIN_S,
    lo: float = 0.0,
    hi: float | None = None,
) -> list[tuple[float, float]]:
    """残す単語の連続区間をマージし、マージン付与＋クランプした時間区間を返す。

    - 連続する残す単語（index が隣接）を1区間にまとめる
    - 各区間の前後に margin を付与
    - ただし前後の（削除された）隣接単語の境界、および [lo, hi] に食い込まない
      範囲でクランプする（削除区間・隣接カットに食い込まない）
    """
    keep = sorted(keep_indices)
    if not keep:
        return []

    # 連続 index をランにまとめる
    runs: list[list[int]] = []
    for idx in keep:
        if runs and idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])

    n = len(words)
    intervals: list[tuple[float, float]] = []
    for run in runs:
        i, j = run[0], run[-1]
        raw_start = float(words[i]["start"])
        raw_end = float(words[j]["end"])

        # 直前の単語（残さない or 存在しない）の end を下限クランプに使う
        prev_bound = float(words[i - 1]["end"]) if i > 0 else lo
        # 直後の単語（残さない or 存在しない）の start を上限クランプに使う
        if j + 1 < n:
            next_bound = float(words[j + 1]["start"])
        else:
            next_bound = hi if hi is not None else (raw_end + margin)

        start = max(raw_start - margin, prev_bound, lo)
        end = raw_end + margin
        end = min(end, next_bound)
        if hi is not None:
            end = min(end, hi)
        if end > start:
            intervals.append((start, end))
    return intervals


def _apply_weights(block, weights):
    """1次元(mono) / 2次元(multi-channel) 双方に weights を掛ける。"""
    if block.ndim == 2:
        return block * weights[:, None]
    return block * weights


def crossfade_concat(chunks: list, fade_samples: int):
    """音声チャンク列を等パワークロスフェードで結合する。

    境界で前チャンクの末尾と次チャンクの先頭を重ね、
    sin/cos の等パワー窓（二乗和=1）でクリックノイズを防ぐ。
    """
    import numpy as np

    valid = [c for c in chunks if len(c) > 0]
    if not valid:
        # 空。呼び出し側で作った配列の形状（mono/2ch）に合わせられないため
        # 空の1次元配列を返す。
        return np.zeros(0, dtype=np.float64)

    out = valid[0].astype(np.float64)
    for nxt in valid[1:]:
        nxt = nxt.astype(np.float64)
        f = min(fade_samples, len(out), len(nxt))
        if f <= 0:
            out = np.concatenate([out, nxt], axis=0)
            continue
        t = (np.arange(f) + 0.5) / f
        fade_out = np.cos(t * np.pi / 2.0)
        fade_in = np.sin(t * np.pi / 2.0)
        tail = out[-f:]
        head = nxt[:f]
        mixed = _apply_weights(tail, fade_out) + _apply_weights(head, fade_in)
        out = np.concatenate([out[:-f], mixed, nxt[f:]], axis=0)
    return out


def _base_name(txt_path: Path) -> str:
    """.sc.txt / .txt を除いたベース名を返す。"""
    name = txt_path.name
    if name.endswith(".sc.txt"):
        return name[: -len(".sc.txt")]
    if name.endswith(".txt"):
        return name[: -len(".txt")]
    return name


def render(txt_path: str, output: str | None = None) -> str:
    """編集後 txt と json を突き合わせて音声を再構成し、出力パスを返す。"""
    import numpy as np
    import soundfile as sf

    txt = Path(txt_path)
    if not txt.exists():
        raise FileNotFoundError(f"編集後テキストが見つかりません: {txt_path}")

    base = _base_name(txt)
    json_path = txt.parent / f"{base}.sc.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f".sc.json が見つかりません（{txt.name} に対応する {json_path.name}）: {json_path}"
        )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    seg_map: dict[str, dict] = {}
    for seg in segments:
        seg_map[seg["id"]] = seg

    source_wav = data.get("source_wav")
    if not source_wav or not Path(source_wav).exists():
        raise FileNotFoundError(f"元 wav が見つかりません: {source_wav}")

    parsed = parse_edited_txt(txt.read_text(encoding="utf-8"))

    # 音声読み込み（元フォーマット維持のため subtype を保持）
    audio, samplerate = sf.read(source_wav, dtype="float64", always_2d=False)
    info = sf.info(source_wav)
    total_samples = len(audio)
    file_length_s = total_samples / samplerate if samplerate else 0.0

    # 各セグメントのマージンクランプ境界を、ソース時間軸（json の並び順）で
    # 隣接する前後セグメントの単語境界から算出する。これにより、あるセグメント
    # 先頭/末尾のマージンが、ソース上で隣り合う別セグメントの単語音声へ食い込む
    # のを防ぐ（設計§8「隣接カット区間に食い込まない範囲で」）。
    # words が空のセグメントは隣接判定から除外する。
    ordered = [s for s in segments if (s.get("words") or [])]
    seg_bounds: dict[str, tuple[float, float]] = {}
    for k, seg in enumerate(ordered):
        prev_words = ordered[k - 1].get("words") if k > 0 else None
        next_words = ordered[k + 1].get("words") if k + 1 < len(ordered) else None
        lo_b = float(prev_words[-1]["end"]) if prev_words else 0.0
        hi_b = float(next_words[0]["start"]) if next_words else file_length_s
        seg_bounds[seg["id"]] = (lo_b, hi_b)

    chunks = []
    all_warnings: list[str] = []
    for seg_id, edited in parsed:
        seg = seg_map.get(seg_id)
        if seg is None:
            raise ValueError(f"json に存在しない ID です: [{seg_id}]")
        words = seg.get("words", []) or []
        word_strs = [w.get("word", "") for w in words]

        keep, warns = surviving_words(word_strs, edited)
        for w in warns:
            all_warnings.append(f"[{seg_id}] {w}")

        lo_b, hi_b = seg_bounds.get(seg_id, (0.0, file_length_s))
        intervals = words_to_intervals(
            words, keep, margin=MARGIN_S, lo=lo_b, hi=hi_b
        )
        for start, end in intervals:
            s0 = max(0, int(round(start * samplerate)))
            s1 = min(total_samples, int(round(end * samplerate)))
            if s1 > s0:
                chunks.append(audio[s0:s1])

    fade_samples = int(round(FADE_S * samplerate))
    result = crossfade_concat(chunks, fade_samples)

    # 出力形状を元音声のチャンネル構成へ合わせる
    if len(result) == 0 and audio.ndim == 2:
        result = np.zeros((0, audio.shape[1]), dtype=np.float64)

    if output is None:
        out_path = txt.parent / f"{base}.edited.wav"
    else:
        out_path = Path(output)

    sf.write(str(out_path), result, samplerate, subtype=info.subtype)

    for w in all_warnings:
        print(f"警告: {w}", file=sys.stderr)

    return str(out_path)
