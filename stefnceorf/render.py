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
from dataclasses import dataclass, field
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
GAP_THRESHOLD_S = 1.2
GAP_MAX_S = 0.7


@dataclass(frozen=True)
class EditStats:
    """編集結果の定量サマリー（render / logic 実行後に表示する）。"""

    source_duration_s: float       # 元音声の長さ（秒）
    output_duration_s: float       # 出力音声の長さ（秒）
    filler_requested: int          # 削除指定された（〔〕残し＋手動削除）フィラー数
    filler_cut: int                # 安全と判定し削除した数
    filler_skipped_unsafe: int     # 音響的に不安全で残置した数
    silence_trim_count: int        # 無音ギャップ切り詰め箇所数
    silence_trim_removed_s: float  # 無音切り詰めの合計秒
    filler_cut_removed_s: float    # フィラーカットで削った合計秒
    lines_deleted_removed_s: float # 行削除で削った合計秒
    lines_total: int               # json のセグメント（行）総数
    lines_deleted: int             # 編集で丸ごと削除された行数
    lines_moved: int               # 並べ替えられた行数
    uts_requested: int = 0         # 削除指定された（〔〕残し／行削除）UTS 行数
    uts_cut: int = 0               # 安全と判定しガード付きカットした数
    uts_skipped_unsafe: int = 0    # ポーズ不足・境界不良で残置した数
    uts_cut_removed_s: float = 0.0 # UTSカットで削った合計秒
    unrecognized_requested: int = 0       # 削除指定された（行削除）未認識区間の数
    unrecognized_cut: int = 0             # 安全と判定しガード付きカットした数
    unrecognized_skipped_unsafe: int = 0  # ポーズ不足・境界不良で残置した数
    unrecognized_cut_removed_s: float = 0.0  # 未認識カットで削った合計秒


@dataclass(frozen=True)
class PlanContext:
    """編集計画の構築に必要な「txt 以外」の入力一式（読み込み結果のキャッシュ）。

    同じ音声・json に対して編集後テキストだけを差し替えて何度も計画を作り直す
    用途（review --serve のトグル即試聴）のため、重い読み込み（wav・json）を
    分離して保持する。
    """

    txt_path: Path
    base_name: str
    json_path: Path
    data: dict
    source_wav: Path
    audio: Any
    audio_info: Any
    samplerate: int
    total_samples: int


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
    silence_trim_boundaries: frozenset[int]
    warnings: list[str]
    stats: EditStats
    # 削除要求されたのに安全に削除できなかった未認識/UTS セグメントの理由。
    # 値は "pause"（元位置のポーズ0.25秒を確保できない）/ "boundary"（残す側の
    # 保持区間に音声が混じる）。安全に削除できた ID は含めない。
    # 表示側（Hlyst）はこの dict を参照する（警告文字列の解析は禁止）。
    guarded_cut_reasons: dict[str, str] = field(default_factory=dict)

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


def rms_dbfs_range(audio, samplerate, start: float, end: float) -> float:
    """区間 [start, end) 秒の RMS を dBFS で返す。範囲はファイル内にクランプ。

    stereo(2D) はチャンネル平均を取ってから RMS を計算する。全ゼロ・空区間は
    -inf 扱い（十分小さい値）とする。dBFS は 20*log10(rms)（フルスケール1.0基準）。
    """
    import numpy as np

    total = len(audio)
    if total == 0 or samplerate <= 0:
        return float("-inf")
    a = int(round(start * samplerate))
    b = int(round(end * samplerate))
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


def boundary_rms(
    audio, samplerate, t: float, half_window_s: float = RMS_SAFE_WINDOW_S
) -> float:
    """時刻 t±half_window_s の RMS を dBFS で返す。範囲はファイル内にクランプ。"""
    return rms_dbfs_range(
        audio, samplerate, t - half_window_s, t + half_window_s
    )


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


UNRECOGNIZED_PAUSE_KEEP_S = 0.25  # 未認識区間/UTS削除後に元位置で残す間の合計（FILLER_PAUSE_KEEP_S と同値）
GUARDED_EDGE_SEARCH_S = 0.75  # カット端から無音開始までに許容する非無音（減衰尾・呼気）の最大長。これを超えて内側にしか無音が無い場合は安全に切れないので保持


def plan_guarded_pause(
    cut_start: float,
    cut_end: float,
    silence_spans: list[tuple[float, float]] | None,
    pause_keep: float = UNRECOGNIZED_PAUSE_KEEP_S,
    edge_search_s: float = GUARDED_EDGE_SEARCH_S,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """削除区間の両端に接する既知無音から pause_keep 秒を残す（純関数）。

    未認識区間・UTS行の削除は、デジタル無音の挿入や別位置のルームトーン借用、
    長いクロスフェードに頼らず「元の録音でその位置が本当に無音だった分だけ」
    残すという方針の一部。容量に数えてよいのは cut_start / cut_end 側から
    見て安全に到達できる既知無音スパンだけで、対象から離れた位置の無音や、
    間に未知音声を挟んだ先の無音は数えない。

    境界は Whisper の単語タイムスタンプなので、実際の発話減衰尾により無音の
    開始/終了が境界からずれる（20〜700ms 程度）。そのため両端とも「端に接する
    スパン」だけでなく、端から edge_search_s 秒だけカット内側まで探索する:
      左スパン = merged を先頭から見て cut_start に届く（end が
        cut_start-TOL 以上の）最初のスパン。その start が
        cut_start+edge_search_s 以下、かつ cut_end+TOL 以下（＝右端の
        手前から始まっている）なら採用、満たさなければその時点で None
        （それより後のスパンはさらに深いので探索しない）。無音が cut_end
        を越えて続くのは正常（既知無音なので容量に数えてよい）ため、
        end が cut_end を超えることは棄却理由にしない。
      右スパン = merged を末尾から見て cut_end に届く（start が
        cut_end+TOL 以下の）最初のスパン。その end が
        cut_end-edge_search_s 以上、かつ cut_start-TOL 以上（＝左端を
        越えて始まっていない）なら採用、満たさなければ同様に None。
    cut_start から左スパンの開始（内側にずれている場合）までの減衰尾部分は
    無音と確認できないため、そのまま出力（cut 対象）に残る。

    silence_spans には UTS 検出と同じ細かいRMS無音（.sc.json の rms_silences）を
    渡す。前後のセグメント境界（発話の端）ではなく無音スパン自体を基準にするのは、
    対象の隣に残余の未認識行がある場合でも、その内部にある無音を使えるように
    するため。

    残す区間は無音スパンの外側端へ接続する:
      左 = [左スパン開始, 左スパン開始 + left_keep]
      右 = [右スパン終端 - right_keep, 右スパン終端]
    その間（余剰の無音＋対象音声）が呼び出し側の exact cut 対象になる。

    配分は pause_keep を半分ずつ割り当て、片側の容量が不足していればその
    不足分をもう片方へ回す。左右の容量合計が pause_keep に満たない場合、
    または左右スパンの外側端の距離が pause_keep 未満（残す間の方が長い）の
    場合は安全に残せないので None を返す。

    最後に、算出した keep 区間を除いた実際のカット区間
    (left_edge+left_keep, right_edge-right_keep) が対象区間
    (cut_start, cut_end) と正の長さで重なるかを確認する。片側にしか無音が
    無い短い gap では左右スパンが同一になり得るが、その場合に対象音声を
    まったく削らない「空カット」が成立してしまうのを防ぐため。
    """
    if not silence_spans:
        return None

    # 隣接・重複する区間を先にマージしておく（SILENCE_EDGE_TOL_S 以内の隙間は結合）
    spans = sorted((float(s), float(e)) for s, e in silence_spans if e > s)
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + SILENCE_EDGE_TOL_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    def _find_left() -> tuple[float, float] | None:
        for s, e in merged:
            if e < cut_start - SILENCE_EDGE_TOL_S:
                continue  # まだ cut_start に届いていない。より後ろを探す
            if s > cut_start + edge_search_s or s > cut_end + SILENCE_EDGE_TOL_S:
                return None  # 深すぎる、または右端を越えて始まっている
            return (s, e)
        return None

    def _find_right() -> tuple[float, float] | None:
        for s, e in reversed(merged):
            if s > cut_end + SILENCE_EDGE_TOL_S:
                continue  # まだ cut_end に届いていない。より前を探す
            if e < cut_end - edge_search_s or e < cut_start - SILENCE_EDGE_TOL_S:
                return None  # 深すぎる、または左端の手前で終わっている
            return (s, e)
        return None

    left_span = _find_left()
    right_span = _find_right()
    if left_span is None or right_span is None:
        return None

    left_edge = left_span[0]
    right_edge = right_span[1]
    left_capacity = left_span[1] - left_span[0]
    right_capacity = right_span[1] - right_span[0]

    if left_capacity + right_capacity < pause_keep - 1e-6:
        return None
    if right_edge - left_edge < pause_keep - 1e-6:
        # 左右が同じ（または近接した）無音スパンで、残す間より短い。
        return None

    half = pause_keep / 2.0
    left_keep = min(half, left_capacity)
    right_keep = min(half, right_capacity)
    # 片側の不足分をもう片方へ回す
    deficit = pause_keep - (left_keep + right_keep)
    if deficit > 1e-9:
        left_keep = min(left_capacity, left_keep + deficit)
        deficit = pause_keep - (left_keep + right_keep)
        if deficit > 1e-9:
            right_keep = min(right_capacity, right_keep + deficit)

    left_keep_end = left_edge + left_keep
    right_keep_start = right_edge - right_keep
    if min(right_keep_start, cut_end) - max(left_keep_end, cut_start) <= 1e-6:
        # 実際に削れる対象音声が残らない（空カットになる）ので不可
        return None

    return (
        (left_edge, left_keep_end),
        (right_keep_start, right_edge),
    )


def guarded_cut_boundaries_are_safe(
    audio, samplerate,
    left_keep: tuple[float, float],
    right_keep: tuple[float, float],
    rms_threshold_db: float,
    window_s: float = RMS_SAFE_WINDOW_S,
) -> bool:
    """guarded cut の実際の切断点が「残す側から見て」無音かを検査する（純関数）。

    plan_guarded_pause が返した左右の保持区間から、切断点は
    left_keep.end（左）と right_keep.start（右）になる。ここで測るのは
    **残す側**の音だけである（削除される対象音声は出力に現れないため、
    その音量で判定すると再配分ケースが必ず不合格になる）。

    - 左切断点 t の評価区間は [max(left_keep.start, t - window_s), t]
    - 右切断点 t の評価区間は [t, min(right_keep.end, t + window_s)]

    つまり窓幅は window_s を上限としつつ、実際に残す既知無音区間へクランプする。
    これにより窓が隣接発話（保持区間の外側）へはみ出すことも無い。評価区間が
    空（保持長 0）なら無音である保証が無いので不合格とする。両側とも
    rms_threshold_db 以下のときだけ True。
    """
    l0, l1 = float(left_keep[0]), float(left_keep[1])
    r0, r1 = float(right_keep[0]), float(right_keep[1])

    l_start = max(l0, l1 - window_s)
    if l1 <= l_start:
        return False
    if rms_dbfs_range(audio, samplerate, l_start, l1) > rms_threshold_db:
        return False

    r_end = min(r1, r0 + window_s)
    if r_end <= r0:
        return False
    return rms_dbfs_range(audio, samplerate, r0, r_end) <= rms_threshold_db


def words_to_intervals(
    words: list[dict],
    keep_indices: set[int],
    margin: float = MARGIN_S,
    tail_margin: float = TAIL_MARGIN_S,
    lo: float = 0.0,
    hi: float | None = None,
    split_gap: float | None = None,
) -> list[tuple[float, float]]:
    """残す単語の連続区間をマージし、マージン付与＋クランプした時間区間を返す。

    - 連続する残す単語（index が隣接）を1区間にまとめる。ただし split_gap 指定時は
      単語間ギャップが split_gap を超える境界で分割する
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
        if (
            runs
            and idx == runs[-1][-1] + 1
            and (
                split_gap is None
                or (
                    float(words[idx]["start"])
                    - float(words[runs[-1][-1]]["end"])
                    <= split_gap
                )
            )
        ):
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


def subtract_spans(
    intervals: list[tuple[float, float]],
    spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """intervals から spans と重なる部分を取り除く（必要なら分割する・純関数）。

    guarded cut（未認識/UTS）の削除区間は、対象音声だけでなく「残す 0.25 秒を
    超えた余剰の無音」も含む。その余剰無音は前後の保持区間（残余の未認識行や
    マージンを伸ばした発話行）の**内部**に入り込むことがあり、区間と区間の
    ギャップを見るだけの plan_output_intervals では削れない。そこで計画区間
    そのものから削除区間を差し引く。

    出力順は入力順を保つ（並べ替え済みの計画にも使える）。空になった区間は落とす。
    """
    if not spans:
        return list(intervals)

    merged: list[tuple[float, float]] = []
    for s, e in sorted((float(s), float(e)) for s, e in spans if e > s):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    out: list[tuple[float, float]] = []
    for start, end in intervals:
        cursor = start
        for s, e in merged:
            if e <= cursor or s >= end:
                continue
            if s > cursor:
                out.append((cursor, min(s, end)))
            cursor = max(cursor, e)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return [(s, e) for s, e in out if e > s]


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
    exact_cut_spans: list[tuple[float, float]] | None = None,
) -> tuple[list[tuple[float, float]], list[bool]]:
    """出力順のソース区間列から、ギャップ保持/切り詰めを適用した出力区間列を返す。

    隣接する区間 (prev, cur) の境界について、ソース時間軸上のギャップ
    g = cur.start - prev.end を次の順に評価する:

    1. g < 0（ソースが逆行 = 並べ替え等）: 従来どおり別区間としてクロスフェード結合
    2. exact_cut_spans のうち (pe, ns) に重なるものがある: guarded cut 分岐。
       未認識区間/UTS の削除で、削除する区間が既に秒単位で確定している
       （plan_guarded_pause が左右へ 0.25 秒のポーズを配分済み）ケース。
       重なる span の [a, b] をそのまま切り落とし（前区間終端を a、次区間始端を
       b にする）単純結合する。ここで追加のポーズ配分はしない。この分岐が
       単語判定より先に来るのは、削除対象の未認識音声が word_spans に残って
       いてもギャップ音声を復活させないため。切り落とした前後に残るギャップは
       分岐 3〜5 で改めて評価する（後続行が丸ごと削除されている場合に、その
       音声や長無音を無条件保持しないため）。同一ギャップに複数の exact span
       があれば左から順に同じ処理を繰り返す。
    3. 間に単語（フィラー以外の削除単語）が挟まる: 従来どおりクロスフェード結合
    4. filler_spans のうち (pe, ns) に重なるものがある: フィラーポーズ分岐。
       安全にカットされたフィラーが挟まる無音ギャップなので、削除後に残す間の
       合計を filler_pause_keep に収める。フィラー span の前後で使える無音量
       （pre_avail / post_avail）に応じ、片側 filler_pause_keep/2 を基本とし、
       片側で余った分は他方の avail 範囲で再配分する（合計 =
       min(pre_avail + post_avail, filler_pause_keep)）。無音中のカットなので
       単純結合（cf_flags False）。
    5. g ≤ gap_threshold: ギャップ音声を保持（連続マージ）
       g > gap_threshold: silence_spans とギャップの共通部分のうち gap_max を超える
       既知無音だけを gap_max に切り詰める。旧JSON・未知区間・短い既知無音は
       発話の可能性があるため保持する。

    戻り値: (区間列, クロスフェードフラグ列)。フラグ列の長さは len(区間列)-1。
    True=クロスフェード結合、False=単純結合（ギャップ切り詰め・フィラーポーズ境界）。
    """
    if not intervals:
        return [], []

    fspans = filler_spans or []
    # exact span は重なり・接触をマージして昇順にしておく（同一ギャップに複数の
    # guarded cut があっても左から順に処理できるようにするため）。
    espans: list[tuple[float, float]] = []
    for span_start, span_end in sorted(
        (float(s), float(e))
        for s, e in (exact_cut_spans or [])
        if float(e) > float(s)
    ):
        if espans and span_start <= espans[-1][1]:
            espans[-1] = (espans[-1][0], max(espans[-1][1], span_end))
        else:
            espans.append((span_start, span_end))

    def _known_silence_overlaps(start: float, end: float) -> list[tuple[float, float]]:
        if not silence_spans or end <= start:
            return []
        merged: list[list[float]] = []
        for span_start, span_end in sorted(
            (max(start, float(s)), min(end, float(e)))
            for s, e in silence_spans
            if float(e) > start and float(s) < end
        ):
            if span_end <= span_start:
                continue
            if merged and span_start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], span_end)
            else:
                merged.append([span_start, span_end])
        return [(span_start, span_end) for span_start, span_end in merged]

    out: list[tuple[float, float]] = []
    cf_flags: list[bool] = []
    pending = [intervals[0][0], intervals[0][1]]
    half = gap_max / 2.0
    fhalf = filler_pause_keep / 2.0
    def _emit(crossfade: bool) -> None:
        """pending を確定出力する。

        空区間（exact カット直後の残余処理で生じうる）は出力へ足さず、結合
        フラグだけを引き継ぐ。クロスフェード要求は安全側なので優先する
        （削除単語の境界がクリック防止フェードと hot 境界警告の対象で
        あり続けるようにするため）。
        """
        if out and pending[1] <= pending[0]:
            if crossfade and cf_flags:
                cf_flags[-1] = True
            return
        out.append((pending[0], pending[1]))
        cf_flags.append(crossfade)

    def _resolve_gap(
        gap_start: float, next_start: float, next_end: float
    ) -> None:
        """保持しないギャップ [gap_start, next_start) を通常規則で処理する。

        分岐 3（削除単語）→ 4（フィラーポーズ）→ 5（無音トリム/保持）。
        next_end は次に保持する区間の暫定終端。exact カット後の残余ギャップも
        この関数を通すので、削除された行の音声や長無音が復活しない。
        """
        if _word_in_gap(word_spans, gap_start, next_start):
            # 3. フィラー以外の削除単語が挟まる → クロスフェード結合
            _emit(True)
            pending[:] = [next_start, next_end]
            return
        overlap = [(s, e) for s, e in fspans if s < next_start and e > gap_start]
        if overlap:
            # 4. フィラーポーズ分岐（無音中カット → 単純結合）
            f_start = min(max(s, gap_start) for s, e in overlap)
            f_end = max(min(e, next_start) for s, e in overlap)
            pre_avail = max(0.0, f_start - gap_start)
            post_avail = max(0.0, next_start - f_end)
            pre_keep = min(pre_avail, fhalf)
            post_keep = min(post_avail, fhalf)
            # 片側の余り（fhalf を使い切れなかった分）を他方の avail 範囲で再配分。
            # 合計は min(pre_avail + post_avail, filler_pause_keep) に収まる。
            leftover = filler_pause_keep - pre_keep - post_keep
            add_pre = min(leftover, pre_avail - pre_keep)
            pre_keep += add_pre
            leftover -= add_pre
            post_keep += min(leftover, post_avail - post_keep)
            pending[1] = gap_start + pre_keep
            _emit(False)
            pending[:] = [next_start - post_keep, next_end]
            return
        # 5. 純無音ギャップ
        g = next_start - gap_start
        if g <= gap_threshold or g <= gap_max:
            pending[1] = next_end
            return

        trimmed = False
        for silence_start, silence_end in _known_silence_overlaps(
            gap_start, next_start
        ):
            silence_duration = silence_end - silence_start
            if silence_duration <= gap_max:
                continue
            pending[1] = silence_start + half
            _emit(False)
            pending[:] = [silence_end - half, next_end]
            trimmed = True
        if not trimmed:
            pending[1] = next_end

    for k in range(1, len(intervals)):
        cur = intervals[k]
        pe = intervals[k - 1][1]
        ns = cur[0]
        if ns - pe < 0:
            # 1. ソース逆行 → クロスフェード結合
            _emit(True)
            pending[:] = [cur[0], cur[1]]
            continue
        exact = [(s, e) for s, e in espans if s < ns and e > pe]
        if exact:
            # 2. guarded cut（未認識/UTS）→ 確定済み区間を切り落として単純結合。
            # 前後に残すポーズは既に intervals 側へ含まれているため再配分しない。
            # exact 区間の前後に残るギャップは通常規則へ回す（切り落として終わりに
            # すると、後続の削除行の音声や長無音がそのまま復活してしまう）。
            gap_start = pe
            for span_start, span_end in exact:
                a = max(pending[0], span_start)
                if a > gap_start:
                    _resolve_gap(gap_start, a, a)
                b = min(cur[1], max(span_end, a))
                pending[1] = max(a, pending[0])
                _emit(False)
                pending[:] = [b, b]
                gap_start = b
            if gap_start < ns:
                _resolve_gap(gap_start, ns, cur[1])
            else:
                pending[1] = cur[1]
            continue
        _resolve_gap(pe, ns, cur[1])
    if out and pending[1] <= pending[0]:
        # 末尾が空区間になる場合はフラグ数の整合を保ったまま落とす
        if cf_flags:
            cf_flags.pop()
    else:
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


def _count_moved_lines(indices: list[int]) -> int:
    """json 順 index 列から「並べ替えで動かした行数」を返す（純関数）。

    最長増加部分列（LIS）は「動かさずに済む行」の最大集合なので、
    全体長 - LIS 長 が最小の並べ替え行数になる。indices は distinct なので
    狭義増加の LIS を bisect で O(n log n) で求める。
    """
    import bisect

    tails: list[int] = []
    for x in indices:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(indices) - len(tails)


def _format_mmss(seconds: float) -> str:
    """秒数を M:SS または H:MM:SS 形式に変換する（秒は floor・ゼロ埋め2桁）。

    循環 import 回避のため transcribe._format_time と同仕様で複製している。
    """
    import math

    total_sec = int(math.floor(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_stats(stats: EditStats, *, gaps_preserved: bool = False) -> list[str]:
    """編集結果統計を表示行リストに整形する（純関数、シンプルな英語）。

    出し分け:
    - Lines 行は削除/移動が1件以上のとき（括弧内は該当項目のみ）
    - Fillers 行は filler_requested > 0 のとき（残置0なら括弧省略）
    - UTS 行は uts_requested > 0 のとき（残置0なら括弧省略、Fillers と同形式）
    - Unrecognized 行は unrecognized_requested > 0 のとき（同形式）
    - Silence 行は常時（1箇所は "1 gap"）
    - Duration 行は常時（gaps_preserved=True のとき ", cuts left as gaps" を付記）
    """
    lines: list[str] = []

    kept = stats.lines_total - stats.lines_deleted
    if stats.lines_deleted + stats.lines_moved >= 1:
        parts: list[str] = []
        if stats.lines_deleted:
            part = f"{stats.lines_deleted} deleted"
            if stats.lines_deleted_removed_s > 0:
                part += f", -{stats.lines_deleted_removed_s:.1f}s"
            parts.append(part)
        if stats.lines_moved:
            parts.append(f"{stats.lines_moved} moved")
        lines.append(
            f"Lines: {kept} of {stats.lines_total} kept ({', '.join(parts)})"
        )

    if stats.filler_requested > 0:
        line = f"Fillers: removed {stats.filler_cut} of {stats.filler_requested}"
        if stats.filler_cut_removed_s > 0:
            line += f", -{stats.filler_cut_removed_s:.1f}s"
        if stats.filler_skipped_unsafe > 0:
            line += f" ({stats.filler_skipped_unsafe} kept: too close to speech)"
        lines.append(line)

    if stats.uts_requested > 0:
        line = f"UTS: cut {stats.uts_cut} of {stats.uts_requested}"
        if stats.uts_cut_removed_s > 0:
            line += f", -{stats.uts_cut_removed_s:.1f}s"
        if stats.uts_skipped_unsafe > 0:
            line += f" ({stats.uts_skipped_unsafe} kept: too close to speech)"
        lines.append(line)

    if stats.unrecognized_requested > 0:
        line = (
            f"Unrecognized: cut {stats.unrecognized_cut} "
            f"of {stats.unrecognized_requested}"
        )
        if stats.unrecognized_cut_removed_s > 0:
            line += f", -{stats.unrecognized_cut_removed_s:.1f}s"
        if stats.unrecognized_skipped_unsafe > 0:
            line += (
                f" ({stats.unrecognized_skipped_unsafe} kept: too close to speech)"
            )
        lines.append(line)

    gap_word = "gap" if stats.silence_trim_count == 1 else "gaps"
    removed = (
        f"-{stats.silence_trim_removed_s:.1f}s"
        if stats.silence_trim_removed_s > 0
        else "0.0s"
    )
    lines.append(
        f"Silence: trimmed {stats.silence_trim_count} {gap_word}, {removed}"
    )

    delta = stats.output_duration_s - stats.source_duration_s
    suffix = ", cuts left as gaps" if gaps_preserved else ""
    lines.append(
        f"Duration: {_format_mmss(stats.source_duration_s)} -> "
        f"{_format_mmss(stats.output_duration_s)} ({delta:+.1f}s{suffix})"
    )
    return lines


def load_plan_context(txt_path: str | Path) -> PlanContext:
    """txt に対応する json と元 wav を読み込み PlanContext を返す。

    編集後テキストそのものは読まない（メモリ上の編集内容から計画を作れるように
    するため）。存在チェックの順序・例外文言は従来の build_edit_plan と同じ。
    """
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
    source_wav = data.get("source_wav")
    if not source_wav or not Path(source_wav).exists():
        raise FileNotFoundError(f"元 wav が見つかりません: {source_wav}")

    # 音声読み込み（元フォーマット維持のため subtype を保持）
    audio, samplerate = sf.read(source_wav, dtype="float64", always_2d=False)
    info = sf.info(source_wav)
    return PlanContext(
        txt_path=txt,
        base_name=base,
        json_path=json_path,
        data=data,
        source_wav=Path(source_wav),
        audio=audio,
        audio_info=info,
        samplerate=samplerate,
        total_samples=len(audio),
    )


def build_edit_plan(
    txt_path: str,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
) -> EditPlan:
    """編集後 txt と json から共有編集計画を構築する。"""
    context = load_plan_context(txt_path)
    return build_edit_plan_from_text(
        context,
        context.txt_path.read_text(encoding="utf-8"),
        gap_threshold=gap_threshold,
        gap_max=gap_max,
    )


def build_edit_plan_from_text(
    context: PlanContext,
    edited_text: str,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
) -> EditPlan:
    """PlanContext とメモリ上の編集後テキストから共有編集計画を構築する。

    ファイルに書き出されていない編集内容（review --serve のトグル状態）からも
    同じ計画を作れるようにするため、build_edit_plan の中核をここへ分離している。
    """
    data = context.data
    segments = data.get("segments", [])
    # 無音リストは用途ごとに使い分ける（混ぜると別機能の挙動が変わる）。
    # - block_silences: フィラーの音響安全判定用（粗い silencedetect パス）
    # - trim_silences: 出力の長無音短縮用（認識用切り詰めと同じ元音源区間）
    # - guard_silences: 未認識/UTS の guarded cut 専用（UTS 検出と同じ細かい
    #   RMS パス）。粗いパスは最小無音長が長く、対象の直近ポーズを拾えない。
    silences = data.get("silences")
    # trim_silences 追加前の旧 json だけ block 用 silences へfallbackする。
    trim_silences = (
        data["trim_silences"] if "trim_silences" in data else silences
    )
    # rms_silences を持たない旧 json は粗い silences へ fallback しない
    # （粗いパスで代用すると誤った位置・長さのポーズを根拠にカットしてしまう）。
    # guarded cut は成立せず reason="pause" の保持になる。再 transcribe で解消する。
    guard_silences = data.get("rms_silences")
    seg_map: dict[str, dict] = {}
    for seg in segments:
        seg_map[seg["id"]] = seg

    parsed = parse_edited_txt(edited_text)

    # フィラー〔〕削除の構造マッチング用に辞書を1回ロードする。
    fillers_set = fillers_mod.load_fillers(data.get("language"))

    audio = context.audio
    samplerate = context.samplerate
    total_samples = context.total_samples
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

    # ---- 未認識区間 / UTS の guarded cut ----
    # 削除要求は「元位置の検出済み無音から合計 0.25 秒のポーズを確保できる」
    # かつ「そのポーズ（残す側）に音声が混じっていない」ときだけ実行する。
    # 足りなければ対象音声を保持し、理由を guarded_cut_reasons へ記録する。
    guarded_cut_reasons: dict[str, str] = {}

    def _guarded_cut_span(
        start: float, end: float
    ) -> tuple[tuple[float, float] | None, str | None]:
        """(切り落とす区間, 保持理由) を返す。カットできるなら理由は None。"""
        keeps = plan_guarded_pause(start, end, guard_silences)
        if keeps is None:
            return None, "pause"
        left_keep, right_keep = keeps
        if not guarded_cut_boundaries_are_safe(
            audio, samplerate, left_keep, right_keep, rms_threshold
        ):
            return None, "boundary"
        # 削除するのは対象音声＋余分な周辺無音。残る元位置無音がちょうど 0.25 秒。
        return (left_keep[1], right_keep[0]), None

    def _guarded_label(seg: dict) -> str:
        if seg.get("reason") == "uts":
            duration = max(
                0.0, float(seg["source_end"]) - float(seg["source_start"])
            )
            return f"未転写発話〔{duration:.1f}秒〕"
        return "未認識区間"

    def _guarded_keep_warning(seg_id: str, seg: dict, reason: str) -> str:
        label = _guarded_label(seg)
        if reason == "pause":
            return (
                f"[{seg_id}] {label}は元位置のポーズを"
                "0.25秒確保できないため残しました"
            )
        return f"[{seg_id}] {label}は前後の音声と連続しているため残しました"

    # 安全にカットされた区間の span（後段の分岐判定に使う）。
    # フィラーは谷スナップ＋ポーズ再配分の filler_spans 経路、未認識/UTS は
    # 秒が確定済みの exact_cut_spans 経路と、渡し先が異なるため別リストで持つ。
    # フィラーはスナップ後の値を持つため、word_spans からの除外は値一致でなく
    # (seg_id, wi) キーで行う（スナップで元 span と一致しなくなるため）。
    filler_cut_spans: list[tuple[float, float]] = []
    uts_cut_spans: list[tuple[float, float]] = []
    safe_filler_keys: set[tuple[str, int]] = set()
    safe_uts_ids: set[str] = set()
    unrecognized_cut_spans: list[tuple[float, float]] = []
    safe_unrecognized_ids: set[str] = set()
    filler_requested = 0        # 削除指定されたフィラー総数（統計用）
    filler_skipped_unsafe = 0   # 音響的に不安全で残置した数（統計用）
    uts_requested = 0           # 削除指定された（〔〕残し／行削除）UTS総数（統計用）
    uts_skipped_unsafe = 0      # 安全に削除できず残置した数（統計用）
    unrecognized_requested = 0      # 行削除指定された未認識区間の数（統計用）
    unrecognized_skipped_unsafe = 0 # 安全に削除できず残置した数（統計用）

    # ---- 行ごと削除された未認識/UTS の guarded 判定（TXT に現れないため先に処理） ----
    # 保持する場合の挿入位置は「JSON 元順で直前にある、TXT に残った行」の直後。
    # 前アンカーが無ければ後アンカーの直前、どちらも無ければ末尾へ置く。
    kept_pos = {seg_id: i for i, (seg_id, _t, _b) in enumerate(parsed)}
    ordered_ids = [seg["id"] for seg, _s, _e in ordered]
    insert_before: dict[int, list[tuple[float, float]]] = {}
    insert_after: dict[int, list[tuple[float, float]]] = {}
    trailing_intervals: list[tuple[float, float]] = []
    guarded_line_ids: set[str] = set()          # 行削除要求された未認識/UTS
    guarded_line_removed_s: dict[str, float] = {}  # 実際に削れた秒（保持なら 0）

    def _place_kept_interval(
        k: int, seg_id: str, interval: tuple[float, float]
    ) -> None:
        prev_anchor = next(
            (ordered_ids[j] for j in range(k - 1, -1, -1)
             if ordered_ids[j] in kept_pos),
            None,
        )
        next_anchor = next(
            (ordered_ids[j] for j in range(k + 1, len(ordered_ids))
             if ordered_ids[j] in kept_pos),
            None,
        )
        if prev_anchor is not None:
            insert_after.setdefault(kept_pos[prev_anchor], []).append(interval)
            if (
                next_anchor is not None
                and kept_pos[next_anchor] < kept_pos[prev_anchor]
            ):
                # 前後アンカーが並べ替えで逆転し一意に挿入できない。安全側へ倒し
                # 前アンカーの直後へ置いたことを知らせる（音声は失わない）。
                all_warnings.append(
                    f"[{seg_id}] 前後の行が並べ替えられているため、"
                    f"残した音声は [{prev_anchor}] の直後に置きました"
                )
        elif next_anchor is not None:
            insert_before.setdefault(kept_pos[next_anchor], []).append(interval)
        else:
            trailing_intervals.append(interval)

    for k, sid in enumerate(ordered_ids):
        if sid in kept_pos:
            continue
        seg = seg_map[sid]
        if seg.get("kind") != "unrecognized":
            continue  # 通常 speech 行の削除は従来どおり無条件
        guarded_line_ids.add(sid)
        guarded_line_removed_s[sid] = 0.0
        start = float(seg["source_start"])
        end = float(seg["source_end"])
        is_uts = seg.get("reason") == "uts"
        if is_uts:
            uts_requested += 1
        else:
            unrecognized_requested += 1
        span, reason = _guarded_cut_span(start, end)
        if span is not None:
            if span[1] > span[0]:
                guarded_line_removed_s[sid] = span[1] - span[0]
                if is_uts:
                    uts_cut_spans.append(span)
                else:
                    unrecognized_cut_spans.append(span)
            (safe_uts_ids if is_uts else safe_unrecognized_ids).add(sid)
            continue
        guarded_cut_reasons[sid] = reason
        all_warnings.append(_guarded_keep_warning(sid, seg, reason))
        if is_uts:
            uts_skipped_unsafe += 1
        else:
            unrecognized_skipped_unsafe += 1
        if end > start:
            _place_kept_interval(k, sid, (start, end))

    for i, (seg_id, edited, brackets) in enumerate(parsed):
        plan_intervals.extend(insert_before.get(i, ()))
        seg = seg_map.get(seg_id)
        if seg is None:
            raise ValueError(f"json に存在しない ID です: [{seg_id}]")
        if seg.get("kind") == "unrecognized":
            start = float(seg["source_start"])
            end = float(seg["source_end"])
            is_uts = seg.get("reason") == "uts"
            # 〔〕 が編集後テキストに残っている＝削除指定（UTS/plain 未認識 共通）。
            # 行ごと削除した場合とまったく同じ guarded cut 判定を適用する
            # （元位置ポーズ0.25秒＋残す側の境界RMS）。確保できなければ自動削除
            # せず音声保持（従来の unrecognized 保持と同じ）に倒し警告する。
            if brackets:
                if is_uts:
                    uts_requested += 1
                else:
                    unrecognized_requested += 1
                span, reason = _guarded_cut_span(start, end)
                if span is not None:
                    if span[1] > span[0]:
                        (uts_cut_spans if is_uts else unrecognized_cut_spans).append(
                            span
                        )
                    (safe_uts_ids if is_uts else safe_unrecognized_ids).add(seg_id)
                else:
                    if end > start:
                        plan_intervals.append((start, end))
                    if is_uts:
                        uts_skipped_unsafe += 1
                    else:
                        unrecognized_skipped_unsafe += 1
                    guarded_cut_reasons[seg_id] = reason
                    all_warnings.append(
                        _guarded_keep_warning(seg_id, seg, reason)
                    )
            elif end > start:
                plan_intervals.append((start, end))
            plan_intervals.extend(insert_after.get(i, ()))
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
        filler_requested += len(filler_del)
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
                filler_skipped_unsafe += 1
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
            words, keep, margin=MARGIN_S, lo=lo_seg, hi=hi_seg,
            split_gap=gap_threshold,
        )
        plan_intervals.extend(intervals)
        plan_intervals.extend(insert_after.get(i, ()))

    plan_intervals.extend(trailing_intervals)

    # 全セグメントのコンテンツ区間（ソース時刻・生値）。未認識行が削除された場合も
    # source span を含め、純無音ギャップとして音声が復活しないようにする。安全カット
    # したフィラーは除外する（さもないと分岐2が先に発火してフィラーポーズ分岐に
    # 到達しない）。除外はスナップで値が変わるため (seg_id, wi) キーで行う。
    content_spans: list[tuple[float, float]] = []
    for seg in segments:
        if seg.get("kind") == "unrecognized":
            if seg.get("id") in safe_uts_ids or seg.get("id") in safe_unrecognized_ids:
                continue
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

    # フィラーの精密カットspanはポーズ再配分つきの分岐へ渡す。未認識/UTS の
    # guarded cut は秒が確定済みなので exact_cut_spans（単純結合）へ渡す。
    precise_cut_spans = filler_cut_spans
    exact_cut_spans = uts_cut_spans + unrecognized_cut_spans
    # guarded cut の削除区間は保持区間の内部（残余の未認識行や、マージンを
    # 伸ばした発話行の中）へ食い込みうるので、計画区間から先に差し引く。
    # ギャップ側の処理（plan_output_intervals の分岐2）だけでは削れない。
    plan_intervals = subtract_spans(plan_intervals, exact_cut_spans)
    out_intervals, cf_flags = plan_output_intervals(
        plan_intervals, content_spans, gap_threshold=gap_threshold, gap_max=gap_max,
        filler_spans=precise_cut_spans,
        silence_spans=trim_silences,
        exact_cut_spans=exact_cut_spans,
    )

    # 無音切り詰めの集計。単純結合（cf_flags False）境界の区間間ギャップが削除秒。
    # ただし精密カットポーズ境界（precise_cut_spans）と guarded cut 境界
    # （exact_cut_spans）は無音切り詰めでないため除外する。
    non_trim_spans = precise_cut_spans + exact_cut_spans
    silence_trim_count = 0
    silence_trim_removed_s = 0.0
    silence_trim_boundaries: set[int] = set()
    for k in range(len(out_intervals) - 1):
        if k < len(cf_flags) and not cf_flags[k]:
            pe, ns = out_intervals[k][1], out_intervals[k + 1][0]
            if not any(s < ns and e > pe for s, e in non_trim_spans):
                silence_trim_boundaries.add(k)
                silence_trim_count += 1
                silence_trim_removed_s += ns - pe

    # 行削除・並べ替えの集計。編集順の seg_id を json 順 index に写像して LIS で算出。
    json_index = {seg["id"]: i for i, seg in enumerate(segments)}
    edited_indices = [json_index[seg_id] for seg_id, _, _ in parsed]

    # フィラーカットで削った合計秒（スナップ後のカット区間長を合算）。
    filler_cut_removed_s = sum(e - s for s, e in filler_cut_spans)

    # 行削除で削った合計秒。各セグメントの実音源占有区間はセグメント切り出しと同じ
    # 境界ロジック（ordered: words 先頭start〜末尾end／unrecognized は source_start〜end）
    # を流用する。編集後 txt に現れない seg_id を削除行として合算する。
    # 未認識/UTS の行削除は guarded 判定に従うので、実際に削れた秒だけを合算する
    # （保持した行は 0 秒。lines_deleted 自体は TXT 上の削除要求数を維持する）。
    seg_span = {seg["id"]: (start, end) for seg, start, end in ordered}
    kept_ids = {seg_id for seg_id, _, _ in parsed}
    lines_deleted_removed_s = 0.0
    for sid, (start, end) in seg_span.items():
        if sid in kept_ids:
            continue
        if sid in guarded_line_ids:
            lines_deleted_removed_s += guarded_line_removed_s.get(sid, 0.0)
        else:
            lines_deleted_removed_s += end - start

    stats = EditStats(
        source_duration_s=file_length_s,
        output_duration_s=sum(e - s for s, e in out_intervals),
        filler_requested=filler_requested,
        filler_cut=len(safe_filler_keys),
        filler_skipped_unsafe=filler_skipped_unsafe,
        silence_trim_count=silence_trim_count,
        silence_trim_removed_s=silence_trim_removed_s,
        filler_cut_removed_s=filler_cut_removed_s,
        lines_deleted_removed_s=lines_deleted_removed_s,
        lines_total=len(segments),
        lines_deleted=len(segments) - len(parsed),
        lines_moved=_count_moved_lines(edited_indices),
        uts_requested=uts_requested,
        uts_cut=len(safe_uts_ids),
        uts_skipped_unsafe=uts_skipped_unsafe,
        uts_cut_removed_s=sum(e - s for s, e in uts_cut_spans),
        unrecognized_requested=unrecognized_requested,
        unrecognized_cut=len(safe_unrecognized_ids),
        unrecognized_skipped_unsafe=unrecognized_skipped_unsafe,
        unrecognized_cut_removed_s=sum(
            e - s for s, e in unrecognized_cut_spans
        ),
    )

    # 通常削除（クロスフェード結合）境界が発話に食い込んでいないかを警告のみ出す
    # （カットは原則どおり実行）。rms_threshold は上で発話中央値から算出済みの値。
    all_warnings.extend(
        warn_hot_boundaries(audio, samplerate, out_intervals, cf_flags, rms_threshold)
    )

    return EditPlan(
        txt_path=context.txt_path,
        base_name=context.base_name,
        source_wav=context.source_wav,
        audio=audio,
        audio_info=context.audio_info,
        samplerate=samplerate,
        total_samples=total_samples,
        output_intervals=out_intervals,
        crossfade_flags=cf_flags,
        silence_trim_boundaries=frozenset(silence_trim_boundaries),
        warnings=all_warnings,
        stats=stats,
        guarded_cut_reasons=guarded_cut_reasons,
    )


def plan_chunk_bounds(
    plan: EditPlan,
) -> tuple[list[tuple[int, int]], list[bool], list[int]]:
    """EditPlan の出力区間をサンプル境界へ丸め、空区間を除いて返す（純関数）。

    戻り値 (サンプル境界列, クロスフェードフラグ列, 元 output_intervals index 列)。
    フラグ列は crossfade_concat へそのまま渡せる形（長さ = 境界数 - 1）。

    render_plan（実書き出し）と review の写像表がまったく同じ丸めを共有できるよう、
    切り出し境界の計算だけをここに集約している。
    """
    bounds: list[tuple[int, int]] = []
    kept_indices: list[int] = []
    for i, (start, end) in enumerate(plan.output_intervals):
        s0 = max(0, int(round(start * plan.samplerate)))
        s1 = min(plan.total_samples, int(round(end * plan.samplerate)))
        if s1 > s0:
            bounds.append((s0, s1))
            kept_indices.append(i)

    flags: list[bool] = []
    for j in range(1, len(kept_indices)):
        idx = kept_indices[j] - 1
        flags.append(
            plan.crossfade_flags[idx]
            if idx < len(plan.crossfade_flags) else True
        )
    return bounds, flags, kept_indices


def render_plan(plan: EditPlan, output_path: str | Path) -> str:
    """EditPlan の出力区間を元音声から切り出し・結合して wav を書き出す。

    build_edit_plan を呼ばないので、同じ EditPlan からレビュー音声と写像表の
    両方を作る用途（review）でも計画が再計算されない。警告・統計の出力は
    呼び出し元の責務とする。
    """
    import numpy as np
    import soundfile as sf

    bounds, flags, _ = plan_chunk_bounds(plan)
    chunks = [plan.audio[s0:s1] for s0, s1 in bounds]

    fade_samples = int(round(FADE_S * plan.samplerate))
    result = crossfade_concat(chunks, fade_samples, flags if flags else None)

    # 出力形状を元音声のチャンネル構成へ合わせる
    if len(result) == 0 and plan.audio.ndim == 2:
        result = np.zeros((0, plan.audio.shape[1]), dtype=np.float64)

    out_path = Path(output_path)
    sf.write(
        str(out_path), result, plan.samplerate, subtype=plan.audio_info.subtype
    )
    return str(out_path)


def render_plan_window(
    plan: EditPlan, t_center_s: float, pad_s: float
) -> tuple[Any, int]:
    """出力時刻 t_center_s±pad_s の窓だけを実レンダーと同一内容で切り出す。

    戻り値 (窓の音声配列, 窓の開始サンプル位置)。開始位置は 0 でクランプされ、
    終端は出力全長でクランプされるため、返る配列長は 2*pad_s より短くなりうる。

    方式は「先頭からのストリーミング」。crossfade_concat をチャンク順に追体験し、
    まだ後続のクロスフェードで書き換わりうる末尾（最大 fade サンプル）だけを
    状態として保持し、確定した部分は捨てる（窓に重なる分だけ書き出す）。
    「対象の前後1チャンクだけを取り出して結合する」方式では、fade 長未満の短い
    チャンクが連続したときの累積長クランプ f = min(fade, 累積長, 次チャンク長)
    を再現できず全体レンダーとサンプル一致しないため採らない。
    """
    import numpy as np

    bounds, flags, _ = plan_chunk_bounds(plan)
    fade = int(round(FADE_S * plan.samplerate))
    sr = plan.samplerate

    w0 = max(0, int(round((t_center_s - pad_s) * sr)))
    w1 = max(w0, int(round((t_center_s + pad_s) * sr)))

    channels = plan.audio.shape[1] if plan.audio.ndim == 2 else 0
    shape = (w1 - w0, channels) if channels else (w1 - w0,)
    window = np.zeros(shape, dtype=np.float64)
    written_end = 0  # 窓内で実際に埋まった末尾（相対位置）

    def _emit(pos: int, block) -> None:
        """出力位置 pos から始まる確定ブロックのうち窓に重なる分を書き込む。"""
        nonlocal written_end
        a = max(pos, w0)
        b = min(pos + len(block), w1)
        if b <= a:
            return
        window[a - w0: b - w0] = block[a - pos: b - pos]
        written_end = max(written_end, b - w0)

    total = 0          # これまでの出力累積長（crossfade_concat の len(out)）
    tail = None        # out の末尾 min(total, fade) サンプル（未確定領域）
    for k, (s0, s1) in enumerate(bounds):
        chunk = plan.audio[s0:s1].astype(np.float64)
        n = len(chunk)
        if k == 0:
            tail = chunk[:0]
        do_cf = True
        if k > 0:
            do_cf = flags[k - 1] if k - 1 < len(flags) else True
        f = min(fade, total, n) if (k > 0 and do_cf) else 0

        base = total - len(tail)
        if f > 0:
            t = (np.arange(f) + 0.5) / f
            mixed = (
                _apply_weights(tail[len(tail) - f:], np.cos(t * np.pi / 2.0))
                + _apply_weights(chunk[:f], np.sin(t * np.pi / 2.0))
            )
            buf = np.concatenate(
                [tail[: len(tail) - f], mixed, chunk[f:]], axis=0
            )
        else:
            buf = np.concatenate([tail, chunk], axis=0)
        total = base + len(buf)

        keep = min(total, fade)
        cut = len(buf) - keep
        if cut > 0:
            _emit(base, buf[:cut])
        tail = buf[cut:]

        # 未確定領域の先頭が窓を追い越したら以降は窓に影響しない
        if total - len(tail) >= w1:
            tail = tail[:0]
            break

    if tail is not None and len(tail):
        _emit(total - len(tail), tail)

    return window[:written_end], w0


def render(
    txt_path: str,
    output: str | None = None,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
    stats_out: list[str] | None = None,
) -> str:
    """編集後 txt と json を突き合わせて音声を再構成し、出力パスを返す。

    stats_out が渡された場合は format_stats の行をそこへ append するだけで
    print しない（呼び出し元が任意の位置で出力できる）。
    None の場合は警告出力の後に print する（後方互換）。
    """
    plan = build_edit_plan(txt_path, gap_threshold=gap_threshold, gap_max=gap_max)

    if output is None:
        out_path = plan.txt_path.parent / f"{plan.base_name}.edited.wav"
    else:
        out_path = Path(output)

    render_plan(plan, out_path)

    for w in plan.warnings:
        print(f"警告: {w}", file=sys.stderr)

    if stats_out is not None:
        stats_out.extend(format_stats(plan.stats))
    else:
        for line in format_stats(plan.stats):
            print(line)

    return str(out_path)
