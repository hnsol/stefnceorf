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
GAP_MAX_S = 0.7

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
) -> tuple[list[tuple[float, float]], list[bool]]:
    """出力順のソース区間列から、ギャップ保持/切り詰めを適用した出力区間列を返す。

    隣接する区間 (prev, cur) の境界について、ソース時間軸上のギャップ
    g = cur.start - prev.end を評価する:

    - g < 0（ソースが逆行 = 並べ替え等）: 従来どおり別区間としてクロスフェード結合
    - 間に単語（削除単語）が挟まる: 従来どおり別区間としてクロスフェード結合
    - 純粋な無音ギャップ かつ g ≤ gap_threshold: ギャップ音声を保持
      （＝1つの連続区間としてマージ。クロスフェードは入らない）
    - 純粋な無音ギャップ かつ g > gap_threshold: gap_max に切り詰め
      （前側 gap_max/2 ＋ 後側 gap_max/2 を残し中間をカット、単純結合）

    戻り値: (区間列, クロスフェードフラグ列)。フラグ列の長さは len(区間列)-1。
    True=クロスフェード結合、False=単純結合（ギャップ切り詰め境界）。
    """
    if not intervals:
        return [], []

    out: list[tuple[float, float]] = []
    cf_flags: list[bool] = []
    pending = [intervals[0][0], intervals[0][1]]
    half = gap_max / 2.0
    for k in range(1, len(intervals)):
        cur = intervals[k]
        pe = intervals[k - 1][1]
        ns = cur[0]
        g = ns - pe
        eligible = g >= 0 and not _word_in_gap(word_spans, pe, ns)
        if eligible:
            if g <= gap_threshold or g <= gap_max:
                pending[1] = cur[1]
            else:
                pending[1] = pe + half
                out.append((pending[0], pending[1]))
                cf_flags.append(False)
                pending = [ns - half, cur[1]]
        else:
            out.append((pending[0], pending[1]))
            cf_flags.append(True)
            pending = [cur[0], cur[1]]
    out.append((pending[0], pending[1]))
    return out, cf_flags


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


def render(
    txt_path: str,
    output: str | None = None,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
) -> str:
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

    # フィラー〔〕削除の構造マッチング用に辞書を1回ロードする。
    fillers_set = fillers_mod.load_fillers(data.get("language"))

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

    # 全セグメントの単語区間（ソース時刻・生値）。ギャップに削除単語が挟まるかの判定に使う。
    word_spans: list[tuple[float, float]] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            s, e = w.get("start"), w.get("end")
            if s is not None and e is not None:
                word_spans.append((float(s), float(e)))

    plan_intervals: list[tuple[float, float]] = []
    all_warnings: list[str] = []
    for seg_id, edited, brackets in parsed:
        seg = seg_map.get(seg_id)
        if seg is None:
            raise ValueError(f"json に存在しない ID です: [{seg_id}]")
        words = seg.get("words", []) or []
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

        # ポーズ区切り（ブロック）単位にスナップ。block を持たない旧 json は
        # 各単語を独立ブロック扱い（＝従来の単語スナップ）にフォールバック。
        if words and all("block" in w for w in words):
            blocks = [w["block"] for w in words]
        else:
            blocks = list(range(len(words)))
        keep, extra = snap_to_blocks(keep, blocks)
        if extra:
            removed = "".join(word_strs[i] for i in sorted(extra))
            all_warnings.append(
                f"[{seg_id}] ポーズ区切りに合わせて追加削除: {removed!r}"
            )

        lo_b, hi_b = seg_bounds.get(seg_id, (0.0, file_length_s))
        intervals = words_to_intervals(
            words, keep, margin=MARGIN_S, lo=lo_b, hi=hi_b
        )
        plan_intervals.extend(intervals)

    out_intervals, cf_flags = plan_output_intervals(
        plan_intervals, word_spans, gap_threshold=gap_threshold, gap_max=gap_max
    )

    chunks = []
    kept_indices = []
    for i, (start, end) in enumerate(out_intervals):
        s0 = max(0, int(round(start * samplerate)))
        s1 = min(total_samples, int(round(end * samplerate)))
        if s1 > s0:
            chunks.append(audio[s0:s1])
            kept_indices.append(i)

    valid_flags = []
    for j in range(1, len(kept_indices)):
        idx = kept_indices[j] - 1
        valid_flags.append(cf_flags[idx] if idx < len(cf_flags) else True)

    fade_samples = int(round(FADE_S * samplerate))
    result = crossfade_concat(chunks, fade_samples, valid_flags if valid_flags else None)

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
