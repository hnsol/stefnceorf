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
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import fillers as fillers_mod

# transcribe 側と同じマーカー定義
LOW_CONF_MARK = "◆"
FILLER_OPEN = "〔"
FILLER_CLOSE = "〕"
# ブロック（カット可能単位）境界の区切り記号（transcribe と同一）
BLOCK_SEP = "／"

# 単語の生存判定で無視する句読点・記号（diff の曖昧性対策。fillers._STRIP_PUNCT と同系）
_JOIN_PUNCT = "、。，．！？!?…・,."

# 区間前後に付与するマージン（秒）と結合クロスフェード長（秒）
MARGIN_S = 0.02
TAIL_MARGIN_S = 0.2
FADE_S = 0.008

# ギャップ（無音ポーズ）保持・切り詰めのしきい値と上限（秒）
# g ≤ GAP_THRESHOLD_S: そのまま保持 / g > GAP_THRESHOLD_S: GAP_MAX_S に切り詰め
GAP_THRESHOLD_S = 1.5
GAP_MAX_S = 1.0


@dataclass(frozen=True)
class EditPlan:
    """レンダーと外部編集形式出力で共有する編集計画。"""

    txt_path: Path
    base_name: str
    source_wav: Path
    audio: Any
    audio_info: Any
    samplerate: int
    total_samples: int
    output_intervals: list[tuple[float, float]]
    crossfade_flags: list[bool]
    warnings: list[str]

# フィラー精密カットの定数群
FILLER_PAUSE_KEEP_S = 0.25  # フィラー削除後に残す間の合計（日本語の句間ポーズ自然域 0.2〜0.35s の下限寄り）
RMS_SAFE_WINDOW_S = 0.03    # 安全判定の境界±窓（調音結合のスケール 20〜50ms）
RMS_SAFE_DBFS = -40.0       # この RMS 以下なら境界は発話でないとみなす（発話 RMS −25〜−15dBFS の十分下）
SILENCE_EDGE_TOL_S = 0.01   # json silences の端との一致許容（±10ms）

# フィラーカット境界の谷スナップ＋相対RMS閾値の定数群。
# 実データ検証: 絶対閾値 -40dB が録音の発話帯域（中央値 -38dB）とほぼ重なり、
# クランプ済み境界が音の減衰尾に留まって多数のフィラーが誤って unsafe 判定された。
# 対策は (1) 境界を近傍のRMS最小点（無音の谷）へスナップ (2) 閾値を発話レベルからの相対値化。
SNAP_SEARCH_S = 0.10       # 谷スナップの探索半幅（フィラー前後のポーズ・減衰尾のスケール）
SNAP_STEP_S = 0.005        # 谷探索の刻み
SNAP_RMS_WINDOW_S = 0.015  # 谷探索時のRMS窓半幅（boundary_rms のwindowより狭く、谷を鋭く検出）
RMS_SAFE_REL_DB = 15.0     # 発話中央値からこのdB下を無音側とみなす（実測: 発話-38dB/無音-73dB帯の中間）
MIN_FILLER_CUT_S = 0.03    # スナップ後のカット区間がこれ未満なら意味がないのでカットしない

# `[ID] text` 行のパターン。`[...]` 内の最初の空白区切りトークンをIDとし、以降は無視。
# 新形式 `[0001 0:12] text` でも旧形式 `[0001] text` でも動く。
_LINE_RE = re.compile(r"^\[([^\]\s]+)(?:\s+[^\]]*)?\]\s?(.*)$")
# 〔...〕（フィラー提案）を中身ごと削除するためのパターン
_FILLER_RE = re.compile(re.escape(FILLER_OPEN) + r"[^" + re.escape(FILLER_CLOSE) + r"]*" + re.escape(FILLER_CLOSE))


def _clean_edited_text(text: str) -> tuple[str, list[str]]:
    """編集後テキストから ◆・／ を除去し、〔...〕 を中身ごと削除する。

    戻り値は (クリーン済みテキスト, 残されていた〔〕中身リスト)。〔〕は削除対象
    フィラーの提案なので、テキストに残された〔中身〕は「削除する」意思表示として
    出現順に収集する。中身からは ◆ と ／ を除いた内容を採る（transcribe が付与した
    マーカーを剥がして word 文字列と突き合わせられるようにするため）。

    括弧が外されている（〔〕が無い）箇所は通常の文字として残る。
    ／（ブロック区切り）は残しても消しても差分に影響しないよう除去する。
    """
    contents: list[str] = []
    for m in _FILLER_RE.finditer(text):
        inner = m.group(0)[len(FILLER_OPEN): -len(FILLER_CLOSE)]
        inner = inner.replace(LOW_CONF_MARK, "").replace(BLOCK_SEP, "")
        contents.append(inner)
    text = _FILLER_RE.sub("", text)
    text = text.replace(BLOCK_SEP, "")
    text = text.replace(LOW_CONF_MARK, "")
    return text, contents


def parse_edited_txt(text: str) -> list[tuple[str, str, list[str]]]:
    """編集後 txt をパースして (ID, クリーン済みテキスト, 〔〕中身リスト) の列を返す。

    - 空行（空白のみ含む）は無視する
    - `[ID] text` 形式でない非空行は ValueError
    - ◆ は除去、〔...〕 は削除扱いへ変換する
    - 〔〕の中身は出現順にリストで返す（フィラー削除の構造マッチング用）
    - ID の重複は ValueError
    """
    out: list[tuple[str, str, list[str]]] = []
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
        cleaned, brackets = _clean_edited_text(m.group(2))
        out.append((seg_id, cleaned, brackets))
    return out


def match_filler_deletions(
    words: list[dict],
    bracket_contents: list[str],
    fillers: set[str],
) -> tuple[set[int], list[str]]:
    """編集後テキストに残された〔中身〕列を提案単語へ順序マッチし、
    (フィラー削除確定 word index 集合, 警告列) を返す（純関数）。

    〔〕はフィラー削除の提案マーカー。編集後テキストに〔中身〕が残っている＝
    ユーザーがその提案を受け入れ「削除する」意思表示である。中身を提案単語
    （transcribe が付与した "suggest": True）へ出現順に構造マッチし、対応する
    word index を削除確定として返す。これにより頻出フィラーで文字diffの整列が
    曖昧になっても、削除対象の単語を一意に特定できる。

    候補 index 列は words のうち w.get("suggest") が真のもの（昇順）。1つも無い
    （suggest を持たない旧 json）場合は fillers_mod.is_filler が真のものへ
    フォールバックする。

    マッチは2ポインタの順序マッチ。各 bracket 内容（先頭から）について、未消費の
    候補のうち w["word"].lstrip() == bracket内容 と完全一致する最初のものへ
    マッチして消費する。完全一致が無ければ fillers_mod.normalize_token 同士の
    比較で緩和一致を試す。マッチしなかった bracket は警告を返し、削除自体は
    _clean_edited_text で既に除去済みのため従来の文字diffに委ねる。
    """
    warnings: list[str] = []
    if not bracket_contents:
        return set(), warnings

    candidates = [i for i, w in enumerate(words) if w.get("suggest")]
    if not candidates:
        candidates = [
            i for i, w in enumerate(words)
            if fillers_mod.is_filler(w.get("word", ""), fillers)
        ]

    consumed: set[int] = set()
    filler_del: set[int] = set()
    for content in bracket_contents:
        matched: int | None = None
        # まず完全一致（lstrip 後の word 文字列と bracket 内容）
        for i in candidates:
            if i in consumed:
                continue
            if words[i].get("word", "").lstrip() == content:
                matched = i
                break
        # 次に normalize_token 同士の緩和一致
        if matched is None:
            norm_content = fillers_mod.normalize_token(content)
            for i in candidates:
                if i in consumed:
                    continue
                if fillers_mod.normalize_token(words[i].get("word", "")) == norm_content:
                    matched = i
                    break
        if matched is None:
            warnings.append(
                f"〔{content}〕に対応する提案単語が見つかりません（テキストdiffで処理します）"
            )
            continue
        consumed.add(matched)
        filler_del.add(matched)
    return filler_del, warnings


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
        # 単語先頭・末尾の空白（英語トークンの整形上の区切り）と句読点・記号
        # （verbatim転写の「を、」「あの、」等）は diff の曖昧性で削除側に
        # 寄りやすいため、生存判定は本体文字のみで行う。
        # 例: 「…を、」＋「あの、」→「…を、」の編集で、削除が「あの、」でなく
        # 「、あの」に整列しても「を、」の単語を巻き込まない。
        core_positions = [
            pos + k
            for k, ch in enumerate(w)
            if not ch.isspace() and ch not in _JOIN_PUNCT
        ]
        if not core_positions or all(survived[p] for p in core_positions):
            keep.add(wi)
        pos += length
    return keep, warnings


def snap_to_blocks(
    keep: set[int], blocks: list[int]
) -> tuple[set[int], set[int]]:
    """残す単語集合をブロック単位にスナップする（純関数）。

    ブロック内に1単語でも削除（keep に無い）があれば、そのブロック全体を削除
    する。戻り値: (スナップ後の keep, スナップで追加削除された index 集合)。
    """
    members: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        members.setdefault(b, []).append(i)

    new_keep: set[int] = set()
    for b, idxs in members.items():
        if all(i in keep for i in idxs):
            new_keep.update(idxs)
    extra = set(keep) - new_keep
    return new_keep, extra


def snap_with_filler_exemption(
    keep: set[int], blocks: list[int], filler_del: set[int]
) -> tuple[set[int], set[int]]:
    """filler_del をスナップ判定上「残存扱い」にして snap_to_blocks を適用し、
    (filler_del を除いた keep, 巻き込み追加削除 index) を返す（純関数）。

    フィラー精密カット確定分（filler_del）は同ブロックに残す単語があっても
    ブロック全体を巻き込ませないため、スナップ判定上は残存扱いにする。判定後に
    filler_del 自身を keep から除いて返すことで、フィラーだけを精密にカットする。
    filler_del が空なら snap_to_blocks と同値。
    """
    snapped, extra = snap_to_blocks(keep | filler_del, blocks)
    return snapped - filler_del, extra


def boundary_rms(
    audio, samplerate, t: float, half_window_s: float = RMS_SAFE_WINDOW_S
) -> float:
    """時刻 t±half_window_s の RMS を dBFS で返す。範囲はファイル内にクランプ。

    stereo(2D) はチャンネル平均を取ってから RMS を計算する。全ゼロ・空窓は
    -inf 扱い（十分小さい値）とする。dBFS は 20*log10(rms)（フルスケール1.0基準）。
    """
    import numpy as np

    total = len(audio)
    if total == 0 or samplerate <= 0:
        return float("-inf")
    a = int(round((t - half_window_s) * samplerate))
    b = int(round((t + half_window_s) * samplerate))
    a = max(0, a)
    b = min(total, b)
    if b <= a:
        return float("-inf")
    block = audio[a:b]
    if getattr(block, "ndim", 1) == 2:
        block = block.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(block.astype(np.float64)))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


def filler_cut_is_safe(
    audio, samplerate, word: dict,
    silences: list | None,
    rms_threshold_db: float = RMS_SAFE_DBFS,
) -> bool:
    """フィラー単語 word のカット両境界（start / end）が安全なら True。

    各境界 t は次のいずれかで安全とみなす:
    (1) silences のいずれかの区間 [s, e] について
        s-SILENCE_EDGE_TOL_S <= t <= e+SILENCE_EDGE_TOL_S
        （端±10ms 込みで無音に接している）
    (2) boundary_rms(t) <= rms_threshold_db（境界近傍が発話レベルでない）

    silences が None（旧 json）は (2) のみで判定する。start/end のいずれかが
    None なら安全と言えないため False。両境界がともに安全なときだけ True。
    """
    start = word.get("start")
    end = word.get("end")
    if start is None or end is None:
        return False

    def _safe(t: float) -> bool:
        if silences is not None:
            for span in silences:
                s, e = float(span[0]), float(span[1])
                if s - SILENCE_EDGE_TOL_S <= t <= e + SILENCE_EDGE_TOL_S:
                    return True
        return boundary_rms(audio, samplerate, t) <= rms_threshold_db

    return _safe(float(start)) and _safe(float(end))


def speech_median_dbfs(
    audio, samplerate, word_spans: list[tuple[float, float]]
) -> float | None:
    """全単語の中央時刻の boundary_rms の中央値（発話レベルの推定）を返す。

    word_spans は (start, end) 列。長さ 0.05s 未満の単語は調音の途中や境界の
    誤差が支配的で発話レベルの代表になりにくいため除外する。対象が無ければ、
    または全て無音（-inf）で有効値が無ければ None を返す。相対閾値
    (発話中央値 - RMS_SAFE_REL_DB) の基準として使う。
    """
    import numpy as np

    vals: list[float] = []
    for s, e in word_spans:
        if e - s < 0.05:
            continue
        db = boundary_rms(audio, samplerate, (s + e) / 2.0)
        if np.isfinite(db):
            vals.append(db)
    if not vals:
        return None
    return float(np.median(vals))


def snap_to_trough(
    audio, samplerate, t: float, lo: float, hi: float,
    search_s: float = SNAP_SEARCH_S, step_s: float = SNAP_STEP_S,
    window_s: float = SNAP_RMS_WINDOW_S,
) -> tuple[float, float]:
    """[max(lo, t-search_s), min(hi, t+search_s)] を step_s 刻みで走査し、
    boundary_rms(・, half_window_s=window_s) が最小の時刻へスナップする。

    戻り値 (スナップ後時刻, そのRMS dBFS)。範囲が空（lo/hi クランプで下限>上限）
    なら (t, boundary_rms(t)) を返す。狭い RMS 窓で走査するのは、発話→無音→発話の
    谷（減衰尾を過ぎた無音点）を鋭く捉えて境界をそこへ寄せるため。
    """
    a = max(lo, t - search_s)
    b = min(hi, t + search_s)
    if b < a:
        return t, boundary_rms(audio, samplerate, t, half_window_s=window_s)

    best_t = t
    best_rms = float("inf")
    x = a
    while x <= b + 1e-9:
        db = boundary_rms(audio, samplerate, x, half_window_s=window_s)
        if db < best_rms:
            best_rms = db
            best_t = x
        x += step_s
    return best_t, best_rms


def plan_filler_cut(
    audio, samplerate, word: dict,
    lo: float, hi: float,
    silences: list | None,
    rms_threshold_db: float,
) -> tuple[float, float] | None:
    """フィラー word のカット区間を谷スナップ込みで計画する（純関数）。

    start / end それぞれを近傍の RMS 最小点（無音の谷）へスナップし、両端とも
    安全なら (snapped_start, snapped_end) を返す。安全とは各境界 t について:
    (1) silences のいずれかの区間 [s,e] で s-SILENCE_EDGE_TOL_S <= t <=
        e+SILENCE_EDGE_TOL_S（端±許容込みで無音に接する/内部）
    (2) スナップ後 RMS <= rms_threshold_db
    のいずれか。加えて snapped_end - snapped_start >= MIN_FILLER_CUT_S を要求する。
    どれかを満たさない、または start/end が None なら None（カットしない）。

    lo/hi は隣接語に食い込まないための探索クランプ（lo=前の単語の end もしくは
    セグメント下限、hi=次の単語の start もしくはセグメント上限）。start 側の探索
    範囲は [max(lo, start-search), min(start+search, end)]、end 側は
    [max(end-search, start), min(hi, end+search)] とし互いに逆転しないようにする。
    """
    start = word.get("start")
    end = word.get("end")
    if start is None or end is None:
        return None
    start = float(start)
    end = float(end)

    s_lo = max(lo, start - SNAP_SEARCH_S)
    s_hi = min(start + SNAP_SEARCH_S, end)
    snapped_start, srms = snap_to_trough(audio, samplerate, start, s_lo, s_hi)

    e_lo = max(end - SNAP_SEARCH_S, start)
    e_hi = min(hi, end + SNAP_SEARCH_S)
    snapped_end, erms = snap_to_trough(audio, samplerate, end, e_lo, e_hi)

    def _safe(t: float, rms: float) -> bool:
        if silences is not None:
            for span in silences:
                s, e = float(span[0]), float(span[1])
                if s - SILENCE_EDGE_TOL_S <= t <= e + SILENCE_EDGE_TOL_S:
                    return True
        return rms <= rms_threshold_db

    if not (_safe(snapped_start, srms) and _safe(snapped_end, erms)):
        return None
    if snapped_end - snapped_start < MIN_FILLER_CUT_S:
        return None
    return snapped_start, snapped_end


def words_to_intervals(
    words: list[dict],
    keep_indices: set[int],
    margin: float = MARGIN_S,
    tail_margin: float = TAIL_MARGIN_S,
    lo: float = 0.0,
    hi: float | None = None,
) -> list[tuple[float, float]]:
    """残す単語の連続区間をマージし、マージン付与＋クランプした時間区間を返す。

    - 連続する残す単語（index が隣接）を1区間にまとめる
    - 各区間の前後に margin を付与（後方は tail_margin まで拡張可能）
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
            next_bound = hi if hi is not None else (raw_end + tail_margin)

        start = max(raw_start - margin, prev_bound, lo)
        end = raw_end + min(tail_margin, max(margin, next_bound - raw_end))
        end = min(end, next_bound)
        if hi is not None:
            end = min(end, hi)
        if end > start:
            intervals.append((start, end))
    return intervals


def _word_in_gap(spans: list[tuple[float, float]], pe: float, ns: float,
                 tol: float = 1e-6) -> bool:
    """開区間 (pe, ns) に重なる単語区間が1つでもあれば True。

    削除単語（残さない単語）がギャップに挟まっているかの判定に使う。
    保持/切り出し済みの隣接単語自身は端がマージン分だけ (pe, ns) の外にあるため
    重ならない（このためこの判定は「間に挟まる別単語」だけを検出する）。
    """
    for s, e in spans:
        if s < ns - tol and e > pe + tol:
            return True
    return False


def plan_output_intervals(
    intervals: list[tuple[float, float]],
    word_spans: list[tuple[float, float]],
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
    filler_spans: list[tuple[float, float]] | None = None,
    filler_pause_keep: float = FILLER_PAUSE_KEEP_S,
    silence_spans: list[tuple[float, float]] | None = None,
) -> tuple[list[tuple[float, float]], list[bool]]:
    """出力順のソース区間列から、ギャップ保持/切り詰めを適用した出力区間列を返す。

    隣接する区間 (prev, cur) の境界について、ソース時間軸上のギャップ
    g = cur.start - prev.end を次の順に評価する:

    1. g < 0（ソースが逆行 = 並べ替え等）: 従来どおり別区間としてクロスフェード結合
    2. 間に単語（フィラー以外の削除単語）が挟まる: 従来どおりクロスフェード結合
    3. filler_spans のうち (pe, ns) に重なるものがある: フィラーポーズ分岐。
       安全にカットされたフィラーが挟まる無音ギャップなので、削除後に残す間の
       合計を filler_pause_keep に収める。フィラー span の前後で使える無音量
       （pre_avail / post_avail）に応じ、片側 filler_pause_keep/2 を基本とし、
       片側で余った分は他方の avail 範囲で再配分する（合計 =
       min(pre_avail + post_avail, filler_pause_keep)）。無音中のカットなので
       単純結合（cf_flags False）。
    4. g ≤ gap_threshold: ギャップ音声を保持（連続マージ）
       g > gap_threshold: silence_spans の和集合がギャップ全体を完全被覆する場合だけ
       gap_max に切り詰める。旧JSON・未知区間・部分被覆は発話の可能性があるため
       ギャップ全体を保持する。

    戻り値: (区間列, クロスフェードフラグ列)。フラグ列の長さは len(区間列)-1。
    True=クロスフェード結合、False=単純結合（ギャップ切り詰め・フィラーポーズ境界）。
    """
    if not intervals:
        return [], []

    fspans = filler_spans or []

    def _known_silence_covers(start: float, end: float) -> bool:
        if not silence_spans or end <= start:
            return False
        cursor = start
        merged: list[list[float]] = []
        for span_start, span_end in sorted(
            (float(s), float(e))
            for s, e in silence_spans
            if float(e) > float(s)
        ):
            if merged and span_start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], span_end)
            else:
                merged.append([span_start, span_end])
        for span_start, span_end in merged:
            if span_end <= cursor:
                continue
            if span_start > cursor:
                return False
            cursor = max(cursor, span_end)
            if cursor >= end:
                return True
        return False

    out: list[tuple[float, float]] = []
    cf_flags: list[bool] = []
    pending = [intervals[0][0], intervals[0][1]]
    half = gap_max / 2.0
    fhalf = filler_pause_keep / 2.0
    for k in range(1, len(intervals)):
        cur = intervals[k]
        pe = intervals[k - 1][1]
        ns = cur[0]
        g = ns - pe
        if g < 0:
            # 1. ソース逆行 → クロスフェード結合
            out.append((pending[0], pending[1]))
            cf_flags.append(True)
            pending = [cur[0], cur[1]]
            continue
        if _word_in_gap(word_spans, pe, ns):
            # 2. フィラー以外の削除単語が挟まる → クロスフェード結合
            out.append((pending[0], pending[1]))
            cf_flags.append(True)
            pending = [cur[0], cur[1]]
            continue
        overlap = [(s, e) for s, e in fspans if s < ns and e > pe]
        if overlap:
            # 3. フィラーポーズ分岐（無音中カット → 単純結合）
            f_start = min(max(s, pe) for s, e in overlap)
            f_end = max(min(e, ns) for s, e in overlap)
            pre_avail = max(0.0, f_start - pe)
            post_avail = max(0.0, ns - f_end)
            pre_keep = min(pre_avail, fhalf)
            post_keep = min(post_avail, fhalf)
            # 片側の余り（fhalf を使い切れなかった分）を他方の avail 範囲で再配分。
            # 合計は min(pre_avail + post_avail, filler_pause_keep) に収まる。
            leftover = filler_pause_keep - pre_keep - post_keep
            add_pre = min(leftover, pre_avail - pre_keep)
            pre_keep += add_pre
            leftover -= add_pre
            post_keep += min(leftover, post_avail - post_keep)
            pending[1] = pe + pre_keep
            out.append((pending[0], pending[1]))
            cf_flags.append(False)
            pending = [ns - post_keep, cur[1]]
            continue
        # 4. 純無音ギャップ
        if (
            g <= gap_threshold
            or g <= gap_max
            or not _known_silence_covers(pe, ns)
        ):
            pending[1] = cur[1]
        else:
            pending[1] = pe + half
            out.append((pending[0], pending[1]))
            cf_flags.append(False)
            pending = [ns - half, cur[1]]
    out.append((pending[0], pending[1]))
    return out, cf_flags


def warn_hot_boundaries(
    audio, samplerate,
    out_intervals: list[tuple[float, float]],
    cf_flags: list[bool],
    rms_threshold_db: float,
) -> list[str]:
    """クロスフェード結合される通常削除境界のうち、両側が発話レベルの箇所を警告する（純関数）。

    背景: フィラー削除には plan_filler_cut の音響安全ガード（unsafe なら残す）が
    あるが、通常削除（ユーザーがテキストを直接消した削除）にはガードが無い。通常は
    ブロックスナップでカットが検出済み無音へ揃うため安全だが、block を持たない旧
    .sc.json や --pause-threshold 0 のときはその保証が無い。そこで**警告だけ**出す
    （削除は原則どおり実行。ユーザーが意図した削除を勝手に復活させない）。

    cf_flags[k] が True の境界（= クロスフェード結合 = 通常削除カットまたは並べ替え
    境界）について、前区間終端 t1=out_intervals[k][1] と次区間始端
    t2=out_intervals[k+1][0] の boundary_rms を計算し、どちらかが rms_threshold_db を
    超えていたら警告を1つ追加する。False の境界（無音切り詰め・フィラーポーズ）は
    無音中のカットなので対象外。out_intervals が1個以下なら境界が無いので空リスト。
    """
    warnings: list[str] = []
    for k in range(len(out_intervals) - 1):
        if k >= len(cf_flags) or not cf_flags[k]:
            continue
        t1 = out_intervals[k][1]
        t2 = out_intervals[k + 1][0]
        r1 = boundary_rms(audio, samplerate, t1)
        r2 = boundary_rms(audio, samplerate, t2)
        if r1 > rms_threshold_db or r2 > rms_threshold_db:
            t = t1
            warnings.append(
                f"削除境界が音声と連続しています "
                f"(源音声 {int(t // 60)}:{t % 60:04.1f} 付近)。カットは実行しました"
            )
    return warnings


def _apply_weights(block, weights):
    """1次元(mono) / 2次元(multi-channel) 双方に weights を掛ける。"""
    if block.ndim == 2:
        return block * weights[:, None]
    return block * weights


def crossfade_concat(
    chunks: list, fade_samples: int, crossfade_flags: list[bool] | None = None
):
    """音声チャンク列を結合する。

    crossfade_flags が None または True の境界では等パワークロスフェード
    （sin/cos 窓でクリックノイズ防止）、False の境界では単純結合。
    """
    import numpy as np

    valid = []
    valid_cf: list[bool] = []
    for i, c in enumerate(chunks):
        if len(c) > 0:
            if valid and crossfade_flags is not None:
                valid_cf.append(crossfade_flags[i - 1] if i - 1 < len(crossfade_flags) else True)
            valid.append(c)

    if not valid:
        return np.zeros(0, dtype=np.float64)

    use_flags = valid_cf if crossfade_flags is not None else None

    out = valid[0].astype(np.float64)
    for idx, nxt in enumerate(valid[1:]):
        nxt = nxt.astype(np.float64)
        do_cf = use_flags[idx] if use_flags and idx < len(use_flags) else True
        if not do_cf:
            out = np.concatenate([out, nxt], axis=0)
            continue
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


def build_edit_plan(
    txt_path: str,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
) -> EditPlan:
    """編集後 txt と json から共有編集計画を構築する。"""
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
    # 音響安全判定用の無音区間（旧 json は None → RMS のみで判定）
    silences = data.get("silences")
    # 出力の長無音短縮には認識用切り詰めと同じ元音源区間を使う。
    # trim_silences 追加前の旧 json だけ block 用 silences へfallbackする。
    trim_silences = (
        data["trim_silences"] if "trim_silences" in data else silences
    )
    seg_map: dict[str, dict] = {}
    for seg in segments:
        seg_map[seg["id"]] = seg

    source_wav = data.get("source_wav")
    if not source_wav or not Path(source_wav).exists():
        raise FileNotFoundError(f"元 wav が見つかりません: {source_wav}")

    parsed = parse_edited_txt(txt.read_text(encoding="utf-8"))

    # フィラー〔〕削除の構造マッチング用に辞書を1回ロードする。
    fillers_set = fillers_mod.load_fillers(data.get("language"))

    # 音声読み込み（元フォーマット維持のため subtype を保持）
    audio, samplerate = sf.read(source_wav, dtype="float64", always_2d=False)
    info = sf.info(source_wav)
    total_samples = len(audio)
    file_length_s = total_samples / samplerate if samplerate else 0.0

    # 各セグメントのマージンクランプ境界を、ソース時間軸（json の並び順）で
    # 隣接する前後セグメントの音声境界から算出する。これにより、あるセグメント
    # 先頭/末尾のマージンが、ソース上で隣り合う別セグメントの単語音声へ食い込む
    # のを防ぐ（設計§8「隣接カット区間に食い込まない範囲で」）。
    # 未認識セグメントは words が空でも source_start/source_end を境界に使い、
    # 前後 speech のマージンが未認識音声へ重ならないようにする。
    ordered: list[tuple[dict, float, float]] = []
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
        ordered.append((seg, start, end))
    seg_bounds: dict[str, tuple[float, float]] = {}
    for k, (seg, _start, _end) in enumerate(ordered):
        lo_b = ordered[k - 1][2] if k > 0 else 0.0
        hi_b = ordered[k + 1][1] if k + 1 < len(ordered) else file_length_s
        seg_bounds[seg["id"]] = (lo_b, hi_b)

    # 発話レベル（全単語中央時刻 RMS の中央値）から相対 RMS 閾値を決める。
    # 録音の発話帯域が絶対閾値 -40dB と重なると境界がほぼ全て unsafe になるため、
    # 発話中央値 - RMS_SAFE_REL_DB と -40dB のうち低い（厳しい）方を採る
    # （静かな録音では相対値、大きい録音でも -40dB より緩めない）。
    all_word_spans: list[tuple[float, float]] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            s, e = w.get("start"), w.get("end")
            if s is not None and e is not None:
                all_word_spans.append((float(s), float(e)))
    speech_med = speech_median_dbfs(audio, samplerate, all_word_spans)
    rms_threshold = (
        min(RMS_SAFE_DBFS, speech_med - RMS_SAFE_REL_DB)
        if speech_med is not None else RMS_SAFE_DBFS
    )

    plan_intervals: list[tuple[float, float]] = []
    all_warnings: list[str] = []
    # 安全にカットされたフィラー単語の span（片側ポーズ残しの分岐判定に使う）。
    # スナップ後の値を持つため、word_spans からの除外は値一致でなく (seg_id, wi)
    # キーで行う（スナップで元 span と一致しなくなるため）。
    filler_cut_spans: list[tuple[float, float]] = []
    safe_filler_keys: set[tuple[str, int]] = set()
    for seg_id, edited, brackets in parsed:
        seg = seg_map.get(seg_id)
        if seg is None:
            raise ValueError(f"json に存在しない ID です: [{seg_id}]")
        if seg.get("kind") == "unrecognized":
            start = float(seg["source_start"])
            end = float(seg["source_end"])
            if end > start:
                plan_intervals.append((start, end))
            continue
        # words は json 由来の共有 dict。谷スナップで start/end を書き換えるため
        # 要素ごとにコピーして他所（後段の word_spans 構築等）への副作用を防ぐ。
        words = [dict(w) for w in (seg.get("words", []) or [])]
        word_strs = [w.get("word", "").replace(BLOCK_SEP, "") for w in words]

        # 〔〕削除の構造マッチ: 提案単語へ順序対応させて削除確定 index を得る。
        # これらは diff の両辺から除外し、曖昧な文字整列に依存せず確実に削除する。
        filler_del, fwarns = match_filler_deletions(words, brackets, fillers_set)
        for w in fwarns:
            all_warnings.append(f"[{seg_id}] {w}")

        # フィラー削除確定分を除いた縮約列で文字diffを行い、残す単語を求める。
        idx_map = [i for i in range(len(word_strs)) if i not in filler_del]
        keep_reduced, warns = surviving_words(
            [word_strs[i] for i in idx_map], edited
        )
        # 縮約 index を元 index へ逆写像（フィラー削除分は keep に入らない）
        keep = {idx_map[k] for k in keep_reduced}
        for w in warns:
            all_warnings.append(f"[{seg_id}] {w}")

        # ポーズ区切り（ブロック）単位。block を持たない旧 json は各単語を独立
        # ブロック扱い（＝従来の単語スナップ）にフォールバック。
        if words and all("block" in w for w in words):
            blocks = [w["block"] for w in words]
        else:
            blocks = list(range(len(words)))

        # 4a. 手動削除フィラーの取り込み。〔〕でなく文字を直接消したフィラーも、
        # 同ブロックに残す単語があるならブロック巻き込みでなく精密カット対象にする。
        for wi in range(len(words)):
            if wi in keep or wi in filler_del:
                continue
            if not fillers_mod.is_filler(word_strs[wi], fillers_set):
                continue
            b = blocks[wi]
            if any(kj in keep and blocks[kj] == b for kj in keep):
                filler_del.add(wi)

        # 4c. 音響安全判定＋谷スナップ。カット両境界を近傍の無音の谷へ寄せ、
        # 両端とも安全（無音接触/相対RMS以下）でカット長が十分なフィラーだけを
        # 精密カット対象にする。unsafe なら keep に戻し警告する。探索は隣接語に
        # 食い込まないよう lo=前語 end / hi=次語 start（無ければセグメント境界）で
        # クランプする。
        lo_seg, hi_seg = seg_bounds.get(seg_id, (0.0, file_length_s))
        for wi in sorted(filler_del):
            lo_s = float(words[wi - 1]["end"]) if wi > 0 else lo_seg
            hi_s = float(words[wi + 1]["start"]) if wi + 1 < len(words) else hi_seg
            cut = plan_filler_cut(
                audio, samplerate, words[wi], lo_s, hi_s, silences, rms_threshold
            )
            if cut is not None:
                filler_cut_spans.append(cut)
                safe_filler_keys.add((seg_id, wi))
                # スナップ値を words[wi]（コピー）へ反映し、以降の
                # words_to_intervals の隣接クランプにスナップ境界を使わせる。
                words[wi]["start"], words[wi]["end"] = cut
            else:
                filler_del.discard(wi)
                keep.add(wi)
                all_warnings.append(
                    f"[{seg_id}] フィラー〔{word_strs[wi]}〕は前後の音声と連続しているため残しました"
                )

        # 4b. filler_del をスナップ判定上「残存扱い」にしてブロックスナップ。
        keep, extra = snap_with_filler_exemption(keep, blocks, filler_del)
        if extra:
            removed = "".join(word_strs[i] for i in sorted(extra))
            all_warnings.append(
                f"[{seg_id}] ポーズ区切りに合わせて追加削除: {removed!r}"
            )

        intervals = words_to_intervals(
            words, keep, margin=MARGIN_S, lo=lo_seg, hi=hi_seg
        )
        plan_intervals.extend(intervals)

    # 全セグメントのコンテンツ区間（ソース時刻・生値）。未認識行が削除された場合も
    # source span を含め、純無音ギャップとして音声が復活しないようにする。安全カット
    # したフィラーは除外する（さもないと分岐2が先に発火してフィラーポーズ分岐に
    # 到達しない）。除外はスナップで値が変わるため (seg_id, wi) キーで行う。
    content_spans: list[tuple[float, float]] = []
    for seg in segments:
        if seg.get("kind") == "unrecognized":
            start = float(seg["source_start"])
            end = float(seg["source_end"])
            if end > start:
                content_spans.append((start, end))
            continue
        sid = seg.get("id")
        for wi, w in enumerate(seg.get("words", []) or []):
            if (sid, wi) in safe_filler_keys:
                continue
            s, e = w.get("start"), w.get("end")
            if s is not None and e is not None:
                content_spans.append((float(s), float(e)))

    out_intervals, cf_flags = plan_output_intervals(
        plan_intervals, content_spans, gap_threshold=gap_threshold, gap_max=gap_max,
        filler_spans=filler_cut_spans,
        silence_spans=trim_silences,
    )

    # 通常削除（クロスフェード結合）境界が発話に食い込んでいないかを警告のみ出す
    # （カットは原則どおり実行）。rms_threshold は上で発話中央値から算出済みの値。
    all_warnings.extend(
        warn_hot_boundaries(audio, samplerate, out_intervals, cf_flags, rms_threshold)
    )

    return EditPlan(
        txt_path=txt,
        base_name=base,
        source_wav=Path(source_wav),
        audio=audio,
        audio_info=info,
        samplerate=samplerate,
        total_samples=total_samples,
        output_intervals=out_intervals,
        crossfade_flags=cf_flags,
        warnings=all_warnings,
    )


def render(
    txt_path: str,
    output: str | None = None,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
) -> str:
    """編集後 txt と json を突き合わせて音声を再構成し、出力パスを返す。"""
    import numpy as np
    import soundfile as sf

    plan = build_edit_plan(txt_path, gap_threshold=gap_threshold, gap_max=gap_max)

    chunks = []
    kept_indices = []
    for i, (start, end) in enumerate(plan.output_intervals):
        s0 = max(0, int(round(start * plan.samplerate)))
        s1 = min(plan.total_samples, int(round(end * plan.samplerate)))
        if s1 > s0:
            chunks.append(plan.audio[s0:s1])
            kept_indices.append(i)

    valid_flags = []
    for j in range(1, len(kept_indices)):
        idx = kept_indices[j] - 1
        valid_flags.append(
            plan.crossfade_flags[idx]
            if idx < len(plan.crossfade_flags) else True
        )

    fade_samples = int(round(FADE_S * plan.samplerate))
    result = crossfade_concat(chunks, fade_samples, valid_flags if valid_flags else None)

    # 出力形状を元音声のチャンネル構成へ合わせる
    if len(result) == 0 and plan.audio.ndim == 2:
        result = np.zeros((0, plan.audio.shape[1]), dtype=np.float64)

    if output is None:
        out_path = plan.txt_path.parent / f"{plan.base_name}.edited.wav"
    else:
        out_path = Path(output)

    sf.write(
        str(out_path), result, plan.samplerate, subtype=plan.audio_info.subtype
    )

    for w in plan.warnings:
        print(f"警告: {w}", file=sys.stderr)

    return str(out_path)
