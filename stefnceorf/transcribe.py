"""音声の文字起こし: mlx-whisper で認識し .sc.json / .sc.txt を生成する。"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from . import fillers as fillers_mod

# 低信頼マーカー ◆ を付与する probability の閾値（これ未満で付与）
LOW_CONF_THRESHOLD = 0.5

# 認識用wavの無音切り詰め設定（元wavは不変。Whisper幻覚対策）
# silencedetect の noise 閾値(dB)・最小無音長(秒)、および切り詰め後に残す長さ(秒)
SILENCE_DB = -50  # 発話レベルが低い音源で静かな発話を無音と誤検出しないよう緩めに設定（認識用wav切り詰めのみに使用、render出力には影響しない）
SILENCE_MIN_S = 1.5
SILENCE_KEEP_S = 0.7

# verbatim認識チャンクの目標長と許容幅（秒）
CHUNK_TARGET_S = 300.0
CHUNK_TOL_S = 60.0

# ポーズベース区切り（ブロック単位削除）設定
# この秒数以上の無音を「カット可能なポーズ区切り」としてブロック境界にする
PAUSE_THRESHOLD_S = 0.15

# 無音クランプ後の単語長がこれ未満になる補正は棄却する（秒）
CLAMP_MIN_WORD_S = 0.05

# 繰り返し幻覚セグメントの後処理除去（--verbatim の condition_on_previous_text=True
# で長尺の静音区間から発生する繰り返し幻覚への対策。入力側では完全に防げない）
HALLUC_MIN_WORDS = 5     # セグメント内反復判定の最小語数
HALLUC_TOP_RATIO = 0.7   # 最頻トークンの占有率閾値
HALLUC_RUN = 3           # 同一テキストセグメントの連続数閾値

# 幻覚疑い時に無音をスキップする本家 whisper 由来の抑止しきい値（秒）。
# word_timestamps=True が必要（当ツールは常に満たす）。非verbatim とレスキュー
# 再認識でのみ使う。verbatim 本認識では使わない: whisper の単語異常
# ヒューリスティクスがフィラーを幻覚と誤判定して弾くため（A/B実測で
# フィラー候補 67→4）。
HALLUC_SILENCE_SKIP_S = 2.0

# 時間密度異常による幻覚除去（条件d）。実発話は 5〜7 文字/秒だが幻覚は
# 全単語 zero-duration 等で 200 文字/秒超になる。実発話の 3 倍超をしきい値とする。
HALLUC_DENSITY_CHARS_S = 25.0

# エコー幻覚除去（条件e）。直近の非空raw segmentの近似繰り返しを捕まえる。
HALLUC_ECHO_LOOKBACK = 3         # 比較する非空raw segmentの最大件数
HALLUC_ECHO_PREFIX = 15          # 過去segmentとの共通接頭辞の閾値文字数
HALLUC_ECHO_DENSITY_CHARS_S = 12.0  # エコー判定に用いる文字密度しきい値

# デコード温度のフォールバック列（whisper 既定と同じ）。正常窓は先頭 0.0 の
# 1回デコードで済み追加コストなし。compression_ratio / logprob 異常の窓のみ
# 高温で再試行される。非verbatim とレスキュー再認識でのみ使う。verbatim 本認識
# では使わない: 結果温度>0.5 でプロンプトがリセットされ initial_prompt
# （フィラー誘導文）ごと以降の全窓から消えるため（A/B実測でフィラー候補
# 67→4）。verbatim は temperature=0 固定とし、幻覚は後処理検出＋レスキューで
# 対処する。
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# 幻覚区間の再認識レスキュー窓の最小長（秒）。これ未満の窓は内容なしとみなし
# レスキューを省略する（従来通り除去扱い）。
RESCUE_MIN_WINDOW_S = 0.5

# fresh-context verbatimレスキューの最大チャンク長（秒）。
RESCUE_VERBATIM_CHUNK_S = 90.0

_SILENCE_RE = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")

# 表示用マーカー（render 側で除去する前提）
LOW_CONF_MARK = "◆"
FILLER_OPEN = "〔"
FILLER_CLOSE = "〕"
# ブロック（カット可能単位）境界を示す区切り記号（全角スラッシュ、render 側で除去）
BLOCK_SEP = "／"

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# verbatim（フィラーも転写）モード用モデル。turbo はフィラーを吸収し、
# turbo+condT は繰り返し幻覚が出るため、large+prompt+condT を用いる。
VERBATIM_MODEL = "mlx-community/whisper-large-v3-mlx"

# verbatim時に渡す initial_prompt（フィラー例文。読点付きトークンを誘導する）
FILLER_PROMPT = "えーっと、あのー、そのー、まあ、うーん、なんか、えー、あー。"
FILLER_PROMPT_EN = "Um, uh, er, ah, you know, like."


def _convert_to_16k_mono(input_wav: str) -> str:
    """ffmpeg で入力wavを16kHz mono一時wavに変換し、一時ファイルのパスを返す。"""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="stefnceorf_")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_wav,
        "-ac",
        "1",
        "-ar",
        "16000",
        tmp_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        os.unlink(tmp_path)
        raise RuntimeError(
            "ffmpeg が見つかりません。ffmpeg をインストールしてください。"
        ) from exc
    if proc.returncode != 0:
        os.unlink(tmp_path)
        raise RuntimeError(
            f"ffmpeg による変換に失敗しました (input={input_wav!r}):\n{proc.stderr}"
        )
    return tmp_path


def parse_silence_periods(stderr: str) -> list[tuple[float, float]]:
    """ffmpeg silencedetect の stderr から (silence_start, silence_end) 列を抽出する。

    silence_start と直後の silence_end を1組にする。対になる end が無い
    silence_start（ファイル末尾まで無音等）は無視する。
    """
    periods: list[tuple[float, float]] = []
    pending: float | None = None
    for m in _SILENCE_RE.finditer(stderr):
        kind, val = m.group(1), float(m.group(2))
        if kind == "start":
            pending = val
        else:  # end
            if pending is not None:
                periods.append((pending, val))
                pending = None
    return periods


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    target: float = CHUNK_TARGET_S,
    tol: float = CHUNK_TOL_S,
) -> list[tuple[float, float]]:
    """認識用wavを無音境界優先で連続チャンクへ分割する。"""
    chunks: list[tuple[float, float]] = []
    pos = 0.0
    upper = target + tol
    lower = target - tol

    while duration - pos > upper:
        candidates = []
        for start, end in silences:
            midpoint = (start + end) / 2.0
            if pos + lower <= midpoint <= pos + upper:
                candidates.append((start, end, midpoint))
        if candidates:
            _, _, boundary = min(
                candidates,
                key=lambda item: (
                    -(item[1] - item[0]),
                    abs(item[2] - (pos + target)),
                    item[2],
                ),
            )
        else:
            boundary = pos + upper
        chunks.append((pos, boundary))
        pos = boundary

    chunks.append((pos, duration))
    return chunks


def _offset_segments(segments: list[dict], offset: float) -> list[dict]:
    """segmentとwordの時刻へoffsetを加えた非破壊コピーを返す。"""
    out: list[dict] = []
    for segment in segments:
        shifted = dict(segment)
        for key in ("start", "end"):
            if shifted.get(key) is not None:
                shifted[key] = float(shifted[key]) + offset
        words = []
        for word in segment.get("words", []) or []:
            shifted_word = dict(word)
            for key in ("start", "end"):
                if shifted_word.get(key) is not None:
                    shifted_word[key] = float(shifted_word[key]) + offset
            words.append(shifted_word)
        shifted["words"] = words
        out.append(shifted)
    return out


def build_cuts(
    periods: list[tuple[float, float]], keep: float = SILENCE_KEEP_S
) -> list[tuple[float, float]]:
    """無音区間列から、切り詰めで除去する (cut_start_src, cut_len) 列を返す。

    各無音 [s, e]（長さ d = e - s）について d > keep のときのみ、前 keep/2 と
    後 keep/2 を残して中間 [s+keep/2, e-keep/2] を除去する。start 昇順で返す。
    """
    cuts: list[tuple[float, float]] = []
    half = keep / 2.0
    for s, e in periods:
        d = e - s
        if d <= keep:
            continue
        cuts.append((s + half, d - keep))
    cuts.sort(key=lambda c: c[0])
    return cuts


def rec_to_src(t_rec: float, cuts: list[tuple[float, float]]) -> float:
    """認識用wav時刻 t_rec を元音源時刻へ逆写像する（純関数・単調非減少）。

    cuts は build_cuts の出力（start昇順）。カット点ちょうどの時刻は近傍の
    ソース側境界（カット開始）へ寄せる。
    """
    src = t_rec
    cum = 0.0
    for cs, cl in cuts:
        rec_pos = cs - cum
        if t_rec > rec_pos:
            src += cl
            cum += cl
        else:
            break
    return src


def remap_words(
    words: list[dict], cuts: list[tuple[float, float]]
) -> list[dict]:
    """単語列の start/end を rec_to_src で逆写像し、単調性を保つようクランプする。

    cuts が空なら恒等写像。start ≤ end、直前 end ≤ 次 start を保証する。
    """
    out: list[dict] = []
    last: float | None = None
    for w in words:
        s = w.get("start")
        e = w.get("end")
        if s is not None:
            s = rec_to_src(float(s), cuts)
            if last is not None and s < last:
                s = last
        if e is not None:
            e = rec_to_src(float(e), cuts)
            if s is not None and e < s:
                e = s
        nw = dict(w)
        nw["start"] = s
        nw["end"] = e
        out.append(nw)
        if e is not None:
            last = e
        elif s is not None:
            last = s
    return out


def clamp_words_to_silences(
    words: list[dict],
    silences: list[tuple[float, float]],
    min_word_s: float = CLAMP_MIN_WORD_S,
) -> list[dict]:
    """単語の start/end を無音区間の境界へ縮めて補正した新リストを返す（純関数）。

    remap_words と同様に非破壊で、dict をコピーした新リストを返す。

    背景: Whisper の単語時刻はフィラー等で実発話区間より大きくズレることが
    ある（実測: 「あの」end=2.30s だが実際は 1.20s から無音）。silencedetect
    済みの無音区間で単語境界を内側へ補正する（stable-ts の adjust_by_silence
    相当の自前実装）。

    補正規則（各単語 w について、silences の全区間を評価する。昇順前提でなくてよい）:
    - start が無音 [s, e] の内部（s < start < e、開区間）にあれば new_start = e
    - end が無音 [s, e] の内部（s < end < e）にあれば new_end = s
    - いずれも単語を内側へ縮める方向のみ（拡大はしない）

    ガード:
    - 単語全体が同一無音区間の内部（start も end も同じ無音内）→ 無補正
      （無音誤検出の疑い）
    - 両側補正の結果 new_end - new_start < min_word_s → 補正量（|new−元|）が
      小さい側のみ適用を試し、それでも < min_word_s なら無補正
    - 片側のみの補正でも new_end - new_start < min_word_s になるなら無補正
    - start / end が None → その単語は無補正

    区間は内側に縮むだけなので、remap_words が保証した単調性（前 end ≤ 次 start）
    は崩れない。
    """
    out: list[dict] = []
    for w in words:
        nw = dict(w)
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            out.append(nw)
            continue
        start = float(start)
        end = float(end)

        # 単語全体が同一無音区間の内部 → 無音誤検出の疑いで無補正
        if any(s < start < e and s < end < e for s, e in silences):
            out.append(nw)
            continue

        # start が無音内部なら new_start = その無音の e（最も縮む e を採用）
        new_start = start
        for s, e in silences:
            if s < start < e and e > new_start:
                new_start = e
        # end が無音内部なら new_end = その無音の s（最も縮む s を採用）
        new_end = end
        for s, e in silences:
            if s < end < e and s < new_end:
                new_end = s

        start_changed = new_start != start
        end_changed = new_end != end

        if start_changed and end_changed:
            if new_end - new_start >= min_word_s:
                a_start, a_end = new_start, new_end
            else:
                # 補正量の小さい側のみ適用を試す
                start_corr = new_start - start
                end_corr = end - new_end
                if start_corr <= end_corr:
                    if end - new_start >= min_word_s:
                        a_start, a_end = new_start, end
                    else:
                        a_start, a_end = start, end
                else:
                    if new_end - start >= min_word_s:
                        a_start, a_end = start, new_end
                    else:
                        a_start, a_end = start, end
        elif start_changed:
            if end - new_start >= min_word_s:
                a_start, a_end = new_start, end
            else:
                a_start, a_end = start, end
        elif end_changed:
            if new_end - start >= min_word_s:
                a_start, a_end = start, new_end
            else:
                a_start, a_end = start, end
        else:
            a_start, a_end = start, end

        nw["start"] = a_start
        nw["end"] = a_end
        out.append(nw)
    return out


def _finalize_segment_words(
    seg: dict,
    cuts: list[tuple[float, float]],
    silences: list[tuple[float, float]],
) -> list[dict]:
    """segmentのwordを最終JSONへ出力する元音源時刻へ変換する。"""
    words = [
        {
            "word": word.get("word", ""),
            "start": word.get("start"),
            "end": word.get("end"),
            "probability": word.get("probability"),
        }
        for word in seg.get("words", []) or []
    ]
    words = remap_words(words, cuts)
    return clamp_words_to_silences(words, silences)


def _merge_time_spans(
    spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """有効な時刻区間を和集合へ正規化する（純関数）。"""
    merged: list[list[float]] = []
    for start, end in sorted(
        (float(start), float(end))
        for start, end in spans
        if float(end) > float(start)
    ):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _span_fully_covered(
    start: float,
    end: float,
    spans: list[tuple[float, float]],
) -> bool:
    """[start, end] が区間和集合で完全被覆されるときだけ True。"""
    if end <= start:
        return True
    cursor = start
    for span_start, span_end in _merge_time_spans(spans):
        if span_end <= cursor:
            continue
        if span_start > cursor:
            return False
        cursor = max(cursor, span_end)
        if cursor >= end:
            return True
    return False


def _preserve_unrecognized_gaps(
    segments: list[dict],
    cuts: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    source_duration: float,
    known_silences: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """主認識に無い非無音区間を未認識行として保持する（純関数）。

    speech は最終JSONと同じ元音源時刻へ変換した単語範囲、unrecognized は
    source_start/source_end を被覆範囲とする。中間gapは SILENCE_MIN_S より長い
    場合、先頭・末尾gapは長さを問わず、検出済み無音で完全被覆されない限り
    全体を unrecognized として追加する。部分的な無音被覆は安全側で全体保持する。
    """
    duration = max(0.0, float(source_duration))
    known = silences if known_silences is None else known_silences
    coverage: list[tuple[float, float]] = []
    positioned: list[tuple[float, int, dict]] = []

    for index, seg in enumerate(segments):
        if seg.get("kind") == "unrecognized":
            start = float(seg.get("source_start", 0.0))
            end = float(seg.get("source_end", start))
        else:
            words = _finalize_segment_words(seg, cuts, silences)
            timed = [
                (float(word["start"]), float(word["end"]))
                for word in words
                if word.get("start") is not None
                and word.get("end") is not None
            ]
            if not timed:
                positioned.append((float("inf"), index, seg))
                continue
            start = min(span[0] for span in timed)
            end = max(span[1] for span in timed)

        start = min(max(start, 0.0), duration)
        end = min(max(end, start), duration)
        positioned.append((start, index, seg))
        if end > start:
            coverage.append((start, end))

    added: list[dict] = []
    merged_coverage = _merge_time_spans(coverage)
    cursor = 0.0

    def _add_gap(start: float, end: float, edge: bool) -> None:
        if end <= start or _span_fully_covered(start, end, known):
            return
        if not edge and end - start <= SILENCE_MIN_S:
            return
        added.append(
            {
                "kind": "unrecognized",
                "source_start": start,
                "source_end": end,
                "text": "",
                "words": [],
            }
        )

    for start, end in merged_coverage:
        _add_gap(cursor, start, edge=(cursor == 0.0))
        cursor = max(cursor, end)
    _add_gap(cursor, duration, edge=True)

    for offset, seg in enumerate(added, start=len(segments)):
        positioned.append((float(seg["source_start"]), offset, seg))
    positioned.sort(key=lambda item: (item[0], item[1]))
    return [seg for _, _, seg in positioned]


def assign_blocks(
    words: list[dict],
    silences: list[tuple[float, float]],
    threshold: float,
) -> list[int]:
    """単語列にセグメント内ブロックindex（0始まり）を割り当てて返す（純関数）。

    ブロック＝カット可能な単位。以下のいずれかで単語 i とその前の単語の間を
    ブロック境界とする:
    - `duration >= threshold` の無音期間の中点 m が、その単語対の接合時刻
      `(prev.end + cur.start)/2` に最も近く、かつ m が `[prev.start, cur.end]`
      に収まる（他セグメントの無音を巻き込まないガード）
    - 単語ギャップ `cur.start - prev.end >= threshold`（無音切り詰め箇所は
      remap でギャップが拡大し自然に境界になる）

    `start`/`end` が None の対は境界にしない（安全側）。空リストは `[]`。
    `threshold == 0` は全単語を独立ブロックにする（＝全境界＝旧挙動）。

    無音区間 [s, e] が単一単語の [w.start, w.end] に完全包含される場合
    （`w.start <= s and e <= w.end`、≤判定）はブロック境界候補にしない。
    伸ばし音の途中の息継ぎ等、単語内無音で偽の境界（／）が立つのを防ぐ。
    start/end が None の単語は包含判定に使わない。
    """
    n = len(words)
    if n == 0:
        return []
    if threshold <= 0:
        return list(range(n))

    boundary = [False] * n

    # 各接合点（i-1, i の間）の接合時刻。None 時刻対は None。
    joints: list[float | None] = [None]  # index 0 はダミー
    for i in range(1, n):
        pe = words[i - 1].get("end")
        cs = words[i].get("start")
        if pe is None or cs is None:
            joints.append(None)
            continue
        joints.append((float(pe) + float(cs)) / 2.0)
        # 単語ギャップ境界
        if float(cs) - float(pe) >= threshold:
            boundary[i] = True

    # 各無音期間の中点を最寄りの接合点にマップして境界化
    # 無音が単一単語の [start, end] に完全包含される場合は境界候補にしない。
    # 伸ばし音の途中の息継ぎ等、単語内部に収まる無音で偽のブロック境界（／）が
    # 立つのを防ぐため。start/end が None の単語は包含判定に使わない。
    for s, e in silences:
        if (e - s) < threshold:
            continue
        contained = False
        for w in words:
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                continue
            if float(ws) <= s and e <= float(we):
                contained = True
                break
        if contained:
            continue
        m = (s + e) / 2.0
        best_i: int | None = None
        best_dist: float | None = None
        for i in range(1, n):
            j = joints[i]
            if j is None:
                continue
            ps = words[i - 1].get("start")
            ce = words[i].get("end")
            if ps is None or ce is None:
                continue
            if not (float(ps) <= m <= float(ce)):
                continue
            dist = abs(j - m)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i is not None:
            boundary[best_i] = True

    blocks = [0] * n
    b = 0
    for i in range(1, n):
        if boundary[i]:
            b += 1
        blocks[i] = b
    return blocks


def drop_hallucinations(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """繰り返し幻覚とみられるセグメントを除去し (残す, 除去した) に分ける（純関数）。

    入力は Whisper result 形式のセグメント列（各 {"text", "words":
    [{"word","start","end",...}]}）。remap 後・ID採番前の生セグメントに対して
    呼ぶことを想定する。start/end は認識用wav時刻のままでよい。

    以下のいずれかに該当したセグメントを除去する:
    a. セグメント内反復: 空白除去した単語トークン（w["word"].strip()）が
       HALLUC_MIN_WORDS 個以上あり最頻トークンの割合が HALLUC_TOP_RATIO 以上、
       または3トークン以上が全て同一（「今、今、今」型）。
    b. セグメント列反復: text.strip() が同一の非空セグメントが HALLUC_RUN 個
       以上連続する場合、その連続全部を除去する。既に a/c で除去済みの
       セグメントを間に挟んでも連続とみなす（幻覚は空セグメントと交互に
       出ることがあるため）。交互パターン「ああして、/こうして、」は
       同一連続ではないので残る。
    c. 空セグメント: words が空、text.strip() が空、または記号・句読点のみで
       文字（かな・漢字・英数字）を含まない。
    d. 時間密度異常: トークンが HALLUC_MIN_WORDS 個以上あり、発話時間（先頭単語
       の start 〜 末尾単語の end）が正で文字密度が HALLUC_DENSITY_CHARS_S 超、
       または発話時間が 0 以下（全単語 zero-duration）。実発話 5〜7 文字/秒に
       対し幻覚は 200 文字/秒超になるため。
    e. エコー（直近セグメントの近似繰り返し）: 正規化テキスト（空白・句読点を
       除去）が直近 HALLUC_ECHO_LOOKBACK 件の非空raw segment（除去済みか否かを
       問わない）のいずれかと共通接頭辞 HALLUC_ECHO_PREFIX 文字以上で、かつ
       文字密度が HALLUC_ECHO_DENSITY_CHARS_S 超なら除去。ここで非空とは正規化後
       に1文字以上あること。実発話の言い直し反復は密度条件で誤除去を防ぐ。

    戻り値: (残すセグメント列, 除去したセグメント列)。順序は入力順を保つ。
    """
    n = len(segments)
    drop = [False] * n

    def _norm(text: str) -> str:
        """正規化: 空白と句読点を除去する（エコー接頭辞比較用）。"""
        return re.sub(r"[\s、。，．！？!?…・,.]", "", text or "")

    def _density(seg: dict, tokens: list[str]) -> float | None:
        """文字密度（文字/秒）。発話時間>0 なら密度、<=0 なら None を返す。"""
        start = _seg_first_start(seg)
        end = _seg_last_end(seg)
        if start is None or end is None:
            return None
        dur = end - start
        if dur <= 0:
            return None
        return len("".join(tokens)) / dur

    for i, seg in enumerate(segments):
        words = seg.get("words") or []
        text = (seg.get("text") or "").strip()
        # 条件 c: 空セグメント（記号のみも含む）
        if not words or not text or not re.search(r"\w", text):
            drop[i] = True
            continue
        # 条件 a: セグメント内反復
        tokens = [(w.get("word") or "").strip() for w in words]
        if len(tokens) >= 3 and len(set(tokens)) == 1:
            drop[i] = True
        elif len(tokens) >= HALLUC_MIN_WORDS:
            top = Counter(tokens).most_common(1)[0][1]
            if top / len(tokens) >= HALLUC_TOP_RATIO:
                drop[i] = True
        # 条件 d: 時間密度異常（a に該当しなかった場合に評価）
        if not drop[i] and len(tokens) >= HALLUC_MIN_WORDS:
            dens = _density(seg, tokens)
            if dens is None or dens > HALLUC_DENSITY_CHARS_S:
                drop[i] = True

    # 条件 e: 直近3件の非空raw segment（drop済みを含む）の近似繰り返し。
    # raw_historyには正規化後に1文字以上ある入力textを、drop状態と無関係に積む。
    raw_history: list[str] = []
    for i in range(n):
        cur = _norm(segments[i].get("text") or "")
        if not drop[i]:
            words = segments[i].get("words") or []
            tokens = [(w.get("word") or "").strip() for w in words]
            if len(tokens) >= HALLUC_MIN_WORDS:
                for prev in raw_history[-HALLUC_ECHO_LOOKBACK:]:
                    prefix = 0
                    for a, b in zip(cur, prev):
                        if a != b:
                            break
                        prefix += 1
                    if prefix < HALLUC_ECHO_PREFIX:
                        continue
                    dens = _density(segments[i], tokens)
                    if (
                        dens is not None
                        and dens > HALLUC_ECHO_DENSITY_CHARS_S
                    ):
                        drop[i] = True
                    break
        if cur:
            raw_history.append(cur)

    # 条件 b: 同一テキストの連続（a/c で除去済みのセグメントは連続を分断しない）
    live = [i for i in range(n) if not drop[i]]
    i = 0
    while i < len(live):
        text = (segments[live[i]].get("text") or "").strip()
        j = i
        while j < len(live) and (
            segments[live[j]].get("text") or ""
        ).strip() == text:
            j += 1
        if j - i >= HALLUC_RUN:
            for k in range(i, j):
                drop[live[k]] = True
        i = j

    kept = [seg for i, seg in enumerate(segments) if not drop[i]]
    dropped = [seg for i, seg in enumerate(segments) if drop[i]]
    return kept, dropped


def audio_gap_window(
    prev_kept_end: float | None,
    next_kept_start: float | None,
    wav_duration: float,
) -> tuple[float, float]:
    """幻覚除去グループの保持窓（認識用wav時刻）を求める（純関数）。"""
    start = 0.0 if prev_kept_end is None else float(prev_kept_end)
    end = float(wav_duration) if next_kept_start is None else float(next_kept_start)
    return start, max(start, end)


def rescue_window(
    prev_kept_end: float | None,
    next_kept_start: float | None,
    wav_duration: float,
) -> tuple[float, float] | None:
    """幻覚除去グループの再認識レスキュー窓（認識用wav時刻）を求める（純関数）。

    窓 = 「直前の kept セグメント末尾単語の end」〜「直後の kept セグメント先頭
    単語の start」。先頭グループ（prev_kept_end is None）は 0 を、末尾グループ
    （next_kept_start is None）は wav_duration を境界に使う。この窓なら時刻の
    無い空セグメントも含めて失われた音声を全部カバーし、kept と重複しない。

    窓長が RESCUE_MIN_WINDOW_S 未満なら内容なしとみなし None を返す。
    """
    start, end = audio_gap_window(prev_kept_end, next_kept_start, wav_duration)
    if end - start < RESCUE_MIN_WINDOW_S:
        return None
    return (start, end)


def _seg_first_start(seg: dict) -> float | None:
    """セグメント先頭の start（None でない最初の単語）を返す。無ければ None。"""
    for w in seg.get("words") or []:
        if w.get("start") is not None:
            return float(w["start"])
    return None


def _seg_last_end(seg: dict) -> float | None:
    """セグメント末尾の end（None でない最後の単語）を返す。無ければ None。"""
    for w in reversed(seg.get("words") or []):
        if w.get("end") is not None:
            return float(w["end"])
    return None


def _rescue_kwargs(
    model: str, lang: str | None, verbatim: bool = False
) -> dict:
    """レスキュー再認識用whisper kwargsを返す。"""
    kwargs = dict(
        path_or_hf_repo=model,
        word_timestamps=True,
        language=lang,
        temperature=0 if verbatim else TEMPERATURE_FALLBACK,
        condition_on_previous_text=verbatim,
        no_speech_threshold=0.8,
        compression_ratio_threshold=2.0,
    )
    if verbatim:
        kwargs["initial_prompt"] = (
            FILLER_PROMPT_EN if lang == "en" else FILLER_PROMPT
        )
    else:
        kwargs["hallucination_silence_threshold"] = HALLUC_SILENCE_SKIP_S
    return kwargs


def _valid_rescue(segments: list[dict]) -> bool:
    """レスキュー結果が全件採用可能かを返す。"""
    kept, dropped = drop_hallucinations(segments)
    return bool(kept) and not dropped and any(s.get("words") for s in kept)


def _wav_duration(path: str) -> float:
    """wavの長さ（秒）を返す。全サンプルを読まずヘッダ情報から算出する。"""
    import soundfile as sf

    info = sf.info(path)
    return info.frames / float(info.samplerate)


def _slice_wav(src_wav: str, start: float, end: float) -> str:
    """認識用wavの [start, end] 秒を切り出した一時wavを書き、パスを返す。"""
    import soundfile as sf

    audio, sr = sf.read(src_wav, dtype="float32", always_2d=False)
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    clip = audio[a:b]
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="stefnceorf_rescue_")
    os.close(fd)
    sf.write(out_path, clip, sr)
    return out_path


def _rescue_transcribe(
    src_wav: str,
    start: float,
    end: float,
    model: str,
    lang: str | None,
    verbatim: bool,
) -> list[dict]:
    """レスキュー窓を指定モードで再認識し、全件正常なら返す。

    verbatimは90秒以下のfresh contextへ分割し、各チャンクでプロンプトを再投入
    する。単語時刻はチャンク開始を1回だけ加算して元の認識用wav時刻へ戻す。
    いずれかのチャンクに幻覚・空結果・時刻異常があれば全体を失敗とする。
    """
    import mlx_whisper

    validated: list[dict] = []
    chunks = [(start, end)]
    if verbatim and end - start > RESCUE_VERBATIM_CHUNK_S:
        duration = end - start
        absolute_silences = parse_silence_periods(
            _detect_silence(src_wav, d=0.5)
        )
        relative_silences = [
            (max(0.0, silence_start - start), min(duration, silence_end - start))
            for silence_start, silence_end in absolute_silences
            if silence_end > start and silence_start < end
        ]
        chunks = []
        pos = 0.0
        while duration - pos > RESCUE_VERBATIM_CHUNK_S:
            remaining_silences = [
                (silence_start - pos, silence_end - pos)
                for silence_start, silence_end in relative_silences
                if silence_end > pos
                and (silence_start + silence_end) / 2.0
                <= pos + RESCUE_VERBATIM_CHUNK_S
            ]
            planned = plan_chunks(
                duration - pos,
                remaining_silences,
                target=RESCUE_VERBATIM_CHUNK_S,
                tol=15.0,
            )
            chunk_len = min(RESCUE_VERBATIM_CHUNK_S, planned[0][1])
            chunks.append((start + pos, start + pos + chunk_len))
            pos += chunk_len
        chunks.append((start + pos, end))

    for chunk_start, chunk_end in chunks:
        clip = _slice_wav(src_wav, chunk_start, chunk_end)
        try:
            clip_duration = _wav_duration(clip)
            result = mlx_whisper.transcribe(
                clip, **_rescue_kwargs(model, lang, verbatim)
            )
        finally:
            try:
                os.unlink(clip)
            except OSError:
                pass

        if not isinstance(result, dict):
            return []
        rsegs = result.get("segments")
        if not isinstance(rsegs, list):
            return []
        local_validated: list[dict] = []
        local_last_end: float | None = None
        for seg in rsegs:
            if not isinstance(seg, dict):
                return []
            if not isinstance(seg.get("text"), str):
                return []
            raw_words = seg.get("words") or []
            if not isinstance(raw_words, list) or not raw_words:
                return []
            words: list[dict] = []
            for word in raw_words:
                if not isinstance(word, dict):
                    return []
                if not isinstance(word.get("word"), str):
                    return []
                word_start = word.get("start")
                word_end = word.get("end")
                if word_start is None or word_end is None:
                    return []
                try:
                    word_start = float(word_start)
                    word_end = float(word_end)
                except (TypeError, ValueError):
                    return []
                if (
                    not math.isfinite(word_start)
                    or not math.isfinite(word_end)
                ):
                    return []
                if word_start < 0.0 or word_end > clip_duration:
                    return []
                if (
                    word_end <= word_start
                    or (
                        local_last_end is not None
                        and word_start < local_last_end
                    )
                ):
                    return []
                local_last_end = word_end
                words.append(
                    {**word, "start": word_start, "end": word_end}
                )
            local_validated.append({**seg, "words": words})
        if not _valid_rescue(local_validated):
            return []
        validated.extend(
            {
                **seg,
                "words": [
                    {
                        **word,
                        "start": word["start"] + chunk_start,
                        "end": word["end"] + chunk_start,
                    }
                    for word in seg["words"]
                ],
            }
            for seg in local_validated
        )

    normalized: list[tuple[float, float, dict]] = []
    for seg in validated:
        words: list[dict] = []
        for word in seg.get("words") or []:
            word_start = word["start"]
            word_end = word["end"]
            if word_end <= word_start:
                return []
            words.append(
                {**word, "start": word_start, "end": word_end}
            )
        if not words:
            return []
        words.sort(key=lambda word: (word["start"], word["end"]))
        seg_start = min(word["start"] for word in words)
        seg_end = max(word["end"] for word in words)
        if seg_end <= seg_start:
            return []
        normalized.append(
            (seg_start, seg_end, {**seg, "words": words})
        )
    last_end: float | None = None
    for _, _, seg in normalized:
        for word in seg["words"]:
            if last_end is not None and word["start"] < last_end:
                return []
            last_end = word["end"]
    combined = [seg for _, _, seg in normalized]
    if not _valid_rescue(combined):
        return []
    return combined


def _detect_silence(wav_path: str, d: float = SILENCE_MIN_S) -> str:
    """ffmpeg silencedetect を実行し stderr を返す。失敗時は空文字列。

    d は最小無音長（秒）。ポーズ区切り検出時は短い d で1回だけ実行し、
    結果を用途別（切り詰め用・ブロック境界用）にフィルタして使う。
    """
    cmd = [
        "ffmpeg",
        "-i",
        wav_path,
        "-af",
        f"silencedetect=noise={SILENCE_DB}dB:d={d}",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        return ""
    return proc.stderr or ""


def _write_trimmed_wav(
    src_wav: str, cuts: list[tuple[float, float]]
) -> tuple[str, float]:
    """cuts で指定された区間を除去した認識用wavを生成し (path, 除去秒数) を返す。

    cuts が空なら src_wav をそのまま返す（除去秒数0）。
    """
    if not cuts:
        return src_wav, 0.0

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(src_wav, dtype="float32", always_2d=False)
    parts = []
    prev = 0
    removed = 0.0
    for cs, cl in cuts:
        a = max(prev, int(round(cs * sr)))
        b = min(len(audio), int(round((cs + cl) * sr)))
        if b > a:
            parts.append(audio[prev:a])
            removed += (b - a) / sr
            prev = b
    parts.append(audio[prev:])
    new_audio = np.concatenate(parts) if parts else audio

    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="stefnceorf_trim_")
    os.close(fd)
    sf.write(out_path, new_audio, sr)
    return out_path, removed


def _format_time(seconds: float) -> str:
    """秒数を M:SS または H:MM:SS 形式の文字列に変換する（秒は floor・ゼロ埋め2桁）。"""
    total_sec = int(math.floor(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _seg_id(index: int) -> str:
    """0始まりのindexを4桁ゼロ埋めの1始まりIDに変換する。"""
    return f"{index + 1:04d}"


def suggest_filler_indices(
    words: list[dict], blocks: list[int] | None, fillers: set[str]
) -> set[int]:
    """〔〕提案対象（単独ブロックのフィラー）の word index 集合を返す（純関数）。

    フィラー（fillers_mod.is_filler が真）のうち、単独ブロック（同ブロックに
    他語がない）のものだけを提案対象とする。他語と同ブロックのフィラーは消すと
    巻き込みが起きるため提案しない。

    blocks が None のときは「単独ブロック」条件を常に真とみなす（build_segment_line
    の is_solo=True フォールバックと同じ挙動）。
    """
    out: set[int] = set()
    for idx, w in enumerate(words):
        raw = w.get("word", "")
        if not fillers_mod.is_filler(raw, fillers):
            continue
        is_solo = True
        if blocks is not None:
            b = blocks[idx]
            is_solo = not any(
                blocks[j] == b for j in range(len(blocks)) if j != idx
            )
        if is_solo:
            out.add(idx)
    return out


def build_segment_line(seg_id: str, words: list[dict], fillers: set[str],
                       filler_suggest: bool,
                       blocks: list[int] | None = None,
                       suggest_indices: set[int] | None = None) -> tuple[str, int]:
    """1セグメントの .sc.txt 行文字列とフィラー候補数を組み立てる。

    表示テキストは json の words から再構成する。◆・〔〕・／ を取り除くと
    words の word 文字列の連結に一致する（render の文字diff→単語逆引き用）。

    blocks が渡された場合、ブロック境界（blocks[i] != blocks[i-1]）の直前に
    区切り記号 ／ を挿入する（lead 空白・〔〕の外側、◆の前）。

    suggest_indices が None のときは suggest_filler_indices を内部で計算する。
    渡された場合はその index 集合を〔〕提案対象として用いる（txt の〔〕と json の
    "suggest" を構成的に一致させるため）。filler_suggest=False のときは提案しない。
    """
    if filler_suggest and suggest_indices is None:
        suggest_indices = suggest_filler_indices(words, blocks, fillers)
    parts: list[str] = []
    filler_count = 0
    for idx, w in enumerate(words):
        raw = w.get("word", "")
        core = raw.lstrip()
        lead = raw[: len(raw) - len(core)]
        prob = w.get("probability")

        piece = core
        if filler_suggest and suggest_indices is not None and idx in suggest_indices:
            piece = f"{FILLER_OPEN}{piece}{FILLER_CLOSE}"
            filler_count += 1
        if prob is not None and prob < LOW_CONF_THRESHOLD:
            piece = f"{LOW_CONF_MARK}{piece}"
        segment = lead + piece
        if blocks is not None and idx > 0 and blocks[idx] != blocks[idx - 1]:
            segment = BLOCK_SEP + segment
        parts.append(segment)

    text = "".join(parts)
    if words:
        start = words[0].get("start")
        if start is not None:
            return f"[{seg_id} {_format_time(start)}] {text}", filler_count
    return f"[{seg_id}] {text}", filler_count


def transcribe(
    input_wav: str,
    lang: str | None = "ja",
    model: str | None = None,
    filler_suggest: bool = False,
    pause_threshold: float = PAUSE_THRESHOLD_S,
    verbatim: bool = False,
) -> dict:
    """入力wavを文字起こしし、.sc.json / .sc.txt を生成する。

    model が None（未指定）のときは verbatim に応じて既定モデルを選ぶ:
    verbatim=True なら VERBATIM_MODEL、そうでなければ DEFAULT_MODEL。
    明示指定されたモデルはそのまま尊重する。

    verbatim=True のときはフィラーを転写するため initial_prompt と
    condition_on_previous_text=True を渡す（非verbatim時はプロンプトなし・False）。

    戻り値: {"json_path", "txt_path", "filler_count", "data"}
    """
    import mlx_whisper

    if model is None:
        model = VERBATIM_MODEL if verbatim else DEFAULT_MODEL

    input_path = Path(input_wav)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_wav}")

    tmp_wav = _convert_to_16k_mono(str(input_path))
    # silencedetect を1回だけ実行し、用途別にフィルタする（元wavは不変）。
    # pause_threshold>0 のときはブロック境界検出用に短い d で検出する。
    if pause_threshold and pause_threshold > 0:
        det_d = min(pause_threshold, SILENCE_MIN_S)
    else:
        det_d = SILENCE_MIN_S
    periods = parse_silence_periods(_detect_silence(tmp_wav, d=det_d))
    # 認識用wav切り詰め用: SILENCE_MIN_S 以上の長い無音のみ
    cut_periods = [p for p in periods if (p[1] - p[0]) >= SILENCE_MIN_S]
    # ブロック境界用: pause_threshold 以上の全無音
    if pause_threshold and pause_threshold > 0:
        block_silences = [p for p in periods if (p[1] - p[0]) >= pause_threshold]
    else:
        block_silences = []
    # 認識用wavのみ長い無音を切り詰める。単語時刻は後段で元時刻へ逆写像。
    cuts = build_cuts(cut_periods, keep=SILENCE_KEEP_S)
    trimmed_wav, removed_s = _write_trimmed_wav(tmp_wav, cuts)
    recog_wav = trimmed_wav
    # verbatim では温度フォールバックと無音スキップを無効にする（A/B実測:
    # ep22-1 でフィラー候補 67→3〜4 に激減する破壊的相互作用があった）。
    # - 温度フォールバック: 結果温度>0.5 でプロンプトがリセットされ、
    #   initial_prompt（フィラー誘導文）ごと以降の全窓から消える
    # - hallucination_silence_threshold: whisper の単語異常ヒューリスティクス
    #   （低確率・不自然な長さの語）がフィラーそのものを幻覚と誤判定して弾く
    # verbatim の幻覚対策は後処理（drop_hallucinations 条件a〜e＋レスキュー）に
    # 委ねる。非verbatim とレスキュー再認識はフィラーを転写しないため両方有効。
    whisper_kwargs = dict(
        path_or_hf_repo=model,
        word_timestamps=True,
        language=lang,
        temperature=0 if verbatim else TEMPERATURE_FALLBACK,
        condition_on_previous_text=verbatim,
        no_speech_threshold=0.8,
        compression_ratio_threshold=2.0,
        verbose=False,  # mlx-whisper 組み込みの進捗バー(tqdm)を有効化
    )
    if verbatim:
        whisper_kwargs["initial_prompt"] = (
            FILLER_PROMPT_EN if lang == "en" else FILLER_PROMPT
        )
    else:
        whisper_kwargs["hallucination_silence_threshold"] = HALLUC_SILENCE_SKIP_S
    # 幻覚除去グループを再認識レスキューする間、認識用wav（trimmed_wav）を
    # 生存させる必要があるため、本認識からレスキュー完了までを1つの try で囲う。
    halluc_ranges: list[dict] = []
    dropped_count = 0
    try:
        recog_wav_duration = _wav_duration(recog_wav)
        source_wav_end = rec_to_src(recog_wav_duration, cuts)
        if verbatim:
            wav_duration = recog_wav_duration
            chunk_silences = parse_silence_periods(
                _detect_silence(recog_wav, d=0.5)
            )
            chunks = plan_chunks(
                wav_duration,
                chunk_silences,
                target=CHUNK_TARGET_S,
                tol=CHUNK_TOL_S,
            )
            result = None
            combined_segments: list[dict] = []
            for chunk_start, chunk_end in chunks:
                chunk_wav = _slice_wav(recog_wav, chunk_start, chunk_end)
                try:
                    chunk_result = mlx_whisper.transcribe(
                        chunk_wav, **whisper_kwargs
                    )
                finally:
                    try:
                        os.unlink(chunk_wav)
                    except OSError:
                        pass
                if result is None:
                    result = dict(chunk_result)
                combined_segments.extend(
                    _offset_segments(
                        chunk_result.get("segments", []) or [], chunk_start
                    )
                )
            if result is None:
                result = {"language": lang, "segments": []}
            result["segments"] = combined_segments
        else:
            result = mlx_whisper.transcribe(recog_wav, **whisper_kwargs)

        # 繰り返し幻覚とみられるセグメントをループ前に除去する。
        raw_segments = result.get("segments", []) or []
        kept_segments, dropped_segments = drop_hallucinations(raw_segments)
        dropped_count = len(dropped_segments)

        # 除去グループごとに kept 境界からレスキュー窓を決め、安全設定で再認識して
        # 差し替える。連続する除去を1範囲にまとめ、元音源時刻で報告する。
        if dropped_segments:
            dropped_ids = {id(s) for s in dropped_segments}
            wav_dur = recog_wav_duration
            final_segments: list[dict] = []
            group: list[dict] = []
            last_kept: dict | None = None

            def _process_group(
                group: list[dict],
                prev_kept: dict | None,
                next_kept: dict | None,
            ) -> list[dict]:
                if not group:
                    return []
                # 表示用の先頭サンプルを収集
                sample = ""
                for gseg in group:
                    if not sample:
                        sample = (gseg.get("text") or "").strip()[:20]

                prev_end = _seg_last_end(prev_kept) if prev_kept else None
                next_start = (
                    _seg_first_start(next_kept) if next_kept else None
                )
                win = rescue_window(prev_end, next_start, wav_dur)

                prev_words = (
                    _finalize_segment_words(prev_kept, cuts, periods)
                    if prev_kept else []
                )
                next_words = (
                    _finalize_segment_words(next_kept, cuts, periods)
                    if next_kept else []
                )
                coverage_start = (
                    _seg_last_end({"words": prev_words})
                    if prev_words else None
                )
                coverage_end = (
                    _seg_first_start({"words": next_words})
                    if next_words else None
                )
                coverage_start = (
                    0.0 if coverage_start is None else coverage_start
                )
                coverage_end = (
                    source_wav_end if coverage_end is None else coverage_end
                )
                coverage_end = max(coverage_start, coverage_end)

                rescued_segs: list[dict] = []
                attempts: list[str] = []
                if win is not None:
                    attempts.append("verbatim")
                    rescued_segs = _rescue_transcribe(
                        recog_wav, win[0], win[1], model, lang, True
                    )
                    if not rescued_segs:
                        attempts.append("safe")
                        rescued_segs = _rescue_transcribe(
                            recog_wav, win[0], win[1], model, lang, False
                        )

                # 診断範囲は、JSONで音声を保持する窓境界と一致させる。
                halluc_ranges.append(
                    {
                        "start": coverage_start,
                        "end": coverage_end,
                        "sample": sample,
                        "rescued": bool(rescued_segs),
                        "rescued_segments": len(rescued_segs),
                        "attempts": attempts,
                        "status": (
                            "rescued" if rescued_segs else "unrecognized"
                        ),
                    }
                )

                def _unrecognized(
                    source_start: float, source_end: float
                ) -> dict | None:
                    if source_end <= source_start:
                        return None
                    return {
                        "kind": "unrecognized",
                        "source_start": source_start,
                        "source_end": source_end,
                        "text": sample,
                        "words": [],
                    }

                if not rescued_segs:
                    unrecognized = _unrecognized(
                        coverage_start, coverage_end
                    )
                    return [unrecognized] if unrecognized else []

                covered: list[dict] = []
                cursor = coverage_start
                for rescued in rescued_segs:
                    words = _finalize_segment_words(rescued, cuts, periods)
                    seg_start = min(word["start"] for word in words)
                    seg_end = max(word["end"] for word in words)
                    seg_start = min(
                        max(seg_start, coverage_start), coverage_end
                    )
                    seg_end = min(max(seg_end, seg_start), coverage_end)
                    if seg_start > cursor:
                        unrecognized = _unrecognized(cursor, seg_start)
                        if unrecognized:
                            covered.append(unrecognized)
                    covered.append(rescued)
                    cursor = max(cursor, seg_end)
                if coverage_end > cursor:
                    unrecognized = _unrecognized(cursor, coverage_end)
                    if unrecognized:
                        covered.append(unrecognized)
                return covered

            for seg in raw_segments:
                if id(seg) in dropped_ids:
                    group.append(seg)
                else:
                    final_segments.extend(
                        _process_group(group, last_kept, seg)
                    )
                    group = []
                    final_segments.append(seg)
                    last_kept = seg
            final_segments.extend(_process_group(group, last_kept, None))
            kept_segments = final_segments
    finally:
        for p in {tmp_wav, trimmed_wav}:
            try:
                os.unlink(p)
            except OSError:
                pass

    # --lang 未指定時は Whisper の自動判定言語で辞書を選択する
    fillers = fillers_mod.load_fillers(lang or result.get("language"))

    # 主認識が発話を丸ごと落とした疎なgapも、削除候補にせず音声保持へ倒す。
    # 検出済み無音で完全に説明できるgapだけは未認識行を追加しない。
    kept_segments = _preserve_unrecognized_gaps(
        kept_segments,
        cuts,
        periods,
        source_wav_end,
        known_silences=periods,
    )

    segments_out = []
    lines: list[str] = []
    total_fillers = 0

    for i, seg in enumerate(kept_segments):
        seg_id = _seg_id(i)
        if seg.get("kind") == "unrecognized":
            source_start = float(seg["source_start"])
            source_end = float(seg["source_end"])
            segments_out.append(
                {
                    "id": seg_id,
                    "kind": "unrecognized",
                    "source_start": source_start,
                    "source_end": source_end,
                    "text": seg.get("text", ""),
                    "words": [],
                }
            )
            duration = max(0.0, source_end - source_start)
            lines.append(
                f"[{seg_id} {_format_time(source_start)}] "
                f"⚠ 未認識区間 {duration:.1f}秒（音声保持）"
            )
            continue
        words = _finalize_segment_words(seg, cuts, periods)
        # ポーズベースのブロック割当（各 word に "block" を付与）
        blocks = assign_blocks(words, block_silences, pause_threshold)
        for wi, w in enumerate(words):
            w["block"] = blocks[wi]
        # フィラー〔〕提案対象を計算し、該当 word に "suggest": True を付与する。
        # txt の〔〕と json の "suggest" を同じ集合から構成し一致を保証する。
        sug: set[int] = set()
        if filler_suggest:
            sug = suggest_filler_indices(words, blocks, fillers)
            for wi in sug:
                words[wi]["suggest"] = True
        segments_out.append(
            {
                "id": seg_id,
                "text": seg.get("text", ""),
                "words": words,
            }
        )
        line, fcount = build_segment_line(
            seg_id, words, fillers, filler_suggest, blocks=blocks,
            suggest_indices=sug,
        )
        lines.append(line)
        total_fillers += fcount

    data = {
        "source_wav": str(input_path.resolve()),
        "language": result.get("language", lang),
        "model": model,
        "verbatim": verbatim,
        "pause_threshold": pause_threshold,
        "silence_trim": {
            "count": len(cuts),
            "removed_s": round(removed_s, 3),
        },
        "silences": [[round(s, 3), round(e, 3)] for s, e in block_silences],
        "trim_silences": [
            [round(s, 3), round(e, 3)] for s, e in cut_periods
        ],
        "segments": segments_out,
    }
    if dropped_count:
        data["hallucination_drop"] = {
            "count": dropped_count,
            "ranges": halluc_ranges,
        }

    base = _strip_wav_suffix(input_path)
    json_path = base.parent / f"{base.name}.sc.json"
    txt_path = base.parent / f"{base.name}.sc.txt"

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "filler_count": total_fillers,
        "silence_cut_count": len(cuts),
        "silence_removed_s": removed_s,
        "hallucination_drop_count": dropped_count,
        "hallucination_ranges": halluc_ranges,
        "data": data,
    }


def _strip_wav_suffix(path: Path) -> Path:
    """拡張子(.wav等)を除いたベースパスを返す。"""
    return path.with_suffix("")
