"""音声の文字起こし: mlx-whisper で認識し .sc.json / .sc.txt を生成する。"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from . import fillers as fillers_mod
from .render import RMS_SAFE_DBFS, RMS_SAFE_REL_DB, speech_median_dbfs

# 低信頼マーカー ◆ を付与する probability の閾値（これ未満で付与）
LOW_CONF_THRESHOLD = 0.5

# 認識用wavの無音切り詰め設定（元wavは不変。Whisper幻覚対策）
# silencedetect の noise 閾値(dB)・最小無音長(秒)、および切り詰め後に残す長さ(秒)
SILENCE_DB = -50  # 発話レベルが低い音源で静かな発話を無音と誤検出しないよう緩めに設定（認識用wav切り詰めのみに使用、render出力には影響しない）
SILENCE_MIN_S = 1.2
SILENCE_KEEP_S = 0.7

# verbatim認識チャンクの目標長と許容幅（秒）。turbo は 30s 窓ごとに
# initial_prompt（フィラー誘導文）を効かせられるよう短チャンクにする
# （plan_chunks は無音境界優先のまま。境界が無ければ target+tol で切る）。
CHUNK_TARGET_S = 30.0
CHUNK_TOL_S = 6.0

# ポーズベース区切り（ブロック単位削除）設定
# この秒数以上の無音を「カット可能なポーズ区切り」としてブロック境界にする
PAUSE_THRESHOLD_S = 0.15

# 無音クランプ後の単語長がこれ未満になる補正は棄却する（秒）
CLAMP_MIN_WORD_S = 0.05

# UTS（未転写発話 = Untranscribed Speech）検出の設定。
# Whisper が転写しない 100〜150ms の短い発声（「え」「ん」「あ」、息、相槌）を
# 音響的に見つけて候補行として提示する（自動カットはしない）。
# しきい値は vidtighten（MIT）の実測チューニング値をそのまま用いる。
UTS_MIN_ISLAND_S = 0.10        # 島（非無音区間）として扱う最小長
UTS_MAX_ISLAND_S = 1.2         # これより長い島は通常の発話とみなす
UTS_MIN_ADJACENT_LARGE = 0.25  # 前後の無音のうち長い方に要求する最小長
UTS_MIN_ADJACENT_SMALL = 0.15  # 前後の無音のうち短い方に要求する最小長（口音・息の除外）

# UTS 検出に使う短時間RMS無音検出の窓長（ms）と最小無音長（秒）。
# ffmpeg silencedetect の粗いパス（d=0.15〜1.2s）だとフィラーが発話ブロックへ
# 融合して島の境界が立たないため、専用の細かいパスを別に走らせる。
UTS_RMS_WINDOW_MS = 20
UTS_RMS_MIN_SILENCE_S = 0.10

# UTS 島への誤タイム単語の付け替え設定。
# Whisper は時々、単語の発声そのものは UTS 島の位置にあるのに、タイムスタンプを
# 隣接する無音側へずらして出す（実測: 「という」の実発声は 1751.08-1751.50s だが
# Whisper は 1751.512-1751.666s＝無音側 を付けた）。この場合、島はどの単語にも
# 覆われず UTS 候補として出てしまい、単語自体は無音区間を時刻として持つ。
# 隣接語が十分近く・無音に沈んでいるときだけ島へ付け替え、UTS 候補を消す。
UTS_REATTACH_MAX_GAP_S = 0.3     # 島端と単語の間に許容する最大ギャップ
UTS_REATTACH_MIN_SILENT_FRAC = 0.8  # 単語区間のうち無音に沈んでいるべき最小割合

# 繰り返し幻覚セグメントの後処理除去（--verbatim の短チャンク turbo が
# チャンク境界や静音区間でまれに出す繰り返し・定型幻覚への対策。入力側では
# 完全に防げないため後処理＋レスキューで対処する）
HALLUC_MIN_WORDS = 5     # セグメント内反復判定の最小語数
HALLUC_TOP_RATIO = 0.7   # 最頻トークンの占有率閾値
HALLUC_RUN = 3           # 同一テキストセグメントの連続数閾値

# 幻覚疑い時に無音をスキップする本家 whisper 由来の抑止しきい値（秒）。
# word_timestamps=True が必要（当ツールは常に満たす）。非verbatim とレスキュー
# 再認識でのみ使う。verbatim 本認識では使わない: whisper の単語異常
# ヒューリスティクスがフィラーを幻覚と誤判定して弾くため（A/B実測で
# フィラー候補 67→4）。
HALLUC_SILENCE_SKIP_S = 2.0

# 短チャンク幻覚除去（条件f）に用いる最小文字数。これ未満の短いセグメントは
# 「はい」等の正常な相槌がありうるため密度チェックの対象にしない。
HALLUC_MIN_CHARS = 4

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

# verbatim（フィラーも転写）モード用モデル。2026-07-25 の A/B 実験の結論に基づき
# turbo（DEFAULT_MODEL と同じ）を用いる。turbo＋FILLER_PROMPT＋condT=False＋30s
# チャンク（毎チャンクにプロンプト再投入）が、フィラー捕捉・本文の自然さ・処理
# 速度のすべてで large を上回った（フィラー捕捉 60→88 件、幻覚ほぼゼロ、処理
# 時間半減）。large は逆にフィラーを取りこぼし、長尺静音区間で繰り返し幻覚を
# 出したため本認識では使わない。large はレスキュー窓の局所再認識でのみ使用する
# （下記 RESCUE_MODEL）。
VERBATIM_MODEL = "mlx-community/whisper-large-v3-turbo"

# レスキュー窓（幻覚除去区間）の局所再認識に用いるモデル（large-v3）。狭い窓を
# fresh context で丁寧に取り直す用途では large の精度が有効。
RESCUE_MODEL = "mlx-community/whisper-large-v3-mlx"

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
    """segmentのwordを最終JSONへ出力する元音源時刻へ変換する。

    seg に "_reattached_words"（_reattach_uts_words が付け替え後に書き込む
    キャッシュ）があれば、それをコピーして返す。UTS 島へ付け替えた単語の
    新しい時刻は、cuts/silences からの再計算では復元できない（元の単語時刻を
    上書きした結果のため）。このキャッシュ経由で全呼び出し元（gap保持・
    ブロック割当・最終JSON書き出し）へ同じ補正済み時刻が伝播する。
    呼び出し元ごとの破壊的変更（例: "block" キー追加）が互いに影響しないよう、
    毎回新しい dict のコピーを返す。
    """
    cached = seg.get("_reattached_words")
    if cached is not None:
        return [dict(word) for word in cached]
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


def detect_rms_silences(
    audio,
    samplerate: int,
    word_spans: list[tuple[float, float]],
    window_ms: int = UTS_RMS_WINDOW_MS,
    min_silence_s: float = UTS_RMS_MIN_SILENCE_S,
) -> list[tuple[float, float]]:
    """短時間RMSで無音区間 [(start, end), ...] を返す（純関数）。

    UTS 検出専用の細かい無音パス。ffmpeg silencedetect の粗いパスは最小無音長が
    長くフィラー前後の短いポーズを拾えないため、フィラーが前後の発話ブロックに
    融合してしまい「島」の境界が立たない。そこで window_ms 幅の非重複窓ごとに
    RMS を計算し、連続して閾値を下回る区間のうち min_silence_s 以上のものを
    無音とする。発話の減衰尾を守るための hangover は入れない（フィラー直前の
    100ms 級ポーズを潰さないため）。

    計算式:
      frame_rms[i] = sqrt(mean(x[i*W : (i+1)*W] ** 2))   （W = 窓サンプル数）
      is_silent[i] = 20*log10(frame_rms[i]) < threshold_db

    threshold_db は render 側と同じ適応規約に従う:
      min(RMS_SAFE_DBFS, 発話中央値 - RMS_SAFE_REL_DB)
    絶対閾値 -40dB のみだと、発話帯域が -40dB 近辺に収まる静かな録音で発話まで
    無音と判定されてしまう。逆に相対値だけだと大きい録音で閾値が緩みすぎるため、
    低い（厳しい）方を採る。word_spans から有効な発話中央値が得られない場合
    （単語なし・全て -inf）は RMS_SAFE_DBFS へフォールバックする。

    audio が短すぎて1窓も取れない場合、および samplerate が不正な場合は
    空リストを返す。stereo(2D) はチャンネル平均を取ってから評価する。
    """
    import numpy as np

    if samplerate <= 0:
        return []
    window = int(samplerate * window_ms / 1000)
    if window <= 0:
        return []

    samples = np.asarray(audio)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    n_frames = len(samples) // window
    if n_frames == 0:
        return []

    speech_med = speech_median_dbfs(audio, samplerate, word_spans)
    threshold_db = (
        min(RMS_SAFE_DBFS, speech_med - RMS_SAFE_REL_DB)
        if speech_med is not None else RMS_SAFE_DBFS
    )
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    frames = samples[: n_frames * window].astype(np.float64).reshape(
        n_frames, window
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    is_silent = rms < threshold_amp

    # 無音区間の抽出は np.diff の立ち上がり(+1)/立ち下がり(-1)で一括に行う。
    # 長さ判定は秒ではなく窓数で行う（0.02 秒が二進で厳密に表せず、秒に直すと
    # 境界ちょうどの無音が取りこぼされるため）。
    d = np.diff(is_silent.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    min_frames = max(1, math.ceil(min_silence_s * samplerate / window - 1e-9))
    keep = (ends - starts) >= min_frames
    frame_s = window / float(samplerate)
    return list(
        zip(
            (starts[keep] * frame_s).tolist(),
            (ends[keep] * frame_s).tolist(),
        )
    )


def detect_untranscribed_speech(
    total_duration: float,
    silence_regions: list[tuple[float, float]],
    transcript_segments: list[dict],
    max_duration: float = UTS_MAX_ISLAND_S,
    min_adjacent_large: float = UTS_MIN_ADJACENT_LARGE,
    min_adjacent_small: float = UTS_MIN_ADJACENT_SMALL,
    min_island_duration: float = UTS_MIN_ISLAND_S,
    transcript_words: list[dict] | None = None,
) -> list[tuple[float, float]]:
    """転写に載らなかった短い発声（UTS）の候補区間を返す（純関数）。

    Ported from vidtighten (MIT License, Copyright (c) matsumotoryosuke-dev)
    src/preprod/audio.py — detect_untranscribed_speech。

    Whisper はフィラー（えー、うー、hmm、uh…）を単語列にすら出さないことが
    ある。ここでは「無音に挟まれた非無音区間（島）のうち、どの単語にも覆われて
    いないもの」として検出する。

    設計上の要点（移植元の判断をそのまま踏襲する）:
    - silence_regions は最小無音長 0.1s・hangover なしの細かいパス
      （detect_rms_silences）を前提とする。粗いパスだとフィラーが発話ブロックに
      融合し、segment の時刻に丸ごと覆われてしまう。
    - 重複判定は単語（transcript_words）時刻で行う。segment 時刻は文全体に
      またがり、文境界のフィラーを覆ってしまうため使えない（transcript_words が
      None のときのみ segment 時刻へフォールバックする）。
    - 孤立条件は両側。前後の無音のうち長い方が min_adjacent_large 以上、かつ
      短い方が min_adjacent_small 以上。狭い語間に紛れる口音・息の混入を防ぎつつ、
      実際に間で挟まれた発声だけを拾う。ファイル先頭・末尾は無限長の無音とみなす。

    戻り値: 候補ごとの (start_sec, end_sec)。
    """
    if total_duration <= 0 or not silence_regions:
        return []

    silences = sorted(silence_regions, key=lambda x: x[0])

    # 各島について (start, end, 直前の無音長, 直後の無音長) を作る
    islands: list[tuple[float, float, float, float]] = []
    prev_end = 0.0
    prev_sil_dur = float("inf")   # ファイル先頭は無限長の無音扱い

    for s_start, s_end in silences:
        # 5ms 未満の島は 20ms 窓の量子化誤差と区別できないため候補にしない。
        if s_start > prev_end + 0.005:
            sil_after_dur = s_end - s_start
            islands.append((prev_end, s_start, prev_sil_dur, sil_after_dur))
        prev_sil_dur = s_end - s_start
        prev_end = max(prev_end, s_end)

    # 最後の無音より後ろの島（末尾も同じく 5ms 下限を適用する）
    if prev_end < total_duration - 0.005:
        islands.append((prev_end, total_duration, prev_sil_dur, float("inf")))

    if transcript_words is not None:
        lookup = sorted(transcript_words, key=lambda w: w["start"])
        _w_starts = [w["start"] for w in lookup]

        def overlaps_transcript(start: float, end: float) -> bool:
            # 走査範囲を島の近傍へ絞る（hi: 島の終わり以降に始まる単語は無関係、
            # lo: 島の 1s 前から。単語長は通常 0.5s 以下なので 1s あれば足りる）。
            hi = bisect.bisect_left(_w_starts, end)
            lo = bisect.bisect_left(_w_starts, max(0.0, start - 1.0))
            for i in range(lo, hi):
                # 20ms 許容: 島へ 20ms 以内しか食い込まない単語は「覆っている」
                # とみなさない（Whisper の時刻ジッタの範囲内のため）。
                if lookup[i]["end"] > start + 0.02:
                    return True
            return False
    else:
        trans = sorted(transcript_segments, key=lambda s: s["start"])
        _s_starts = [s["start"] for s in trans]

        def overlaps_transcript(start: float, end: float) -> bool:  # type: ignore[misc]
            # segment は長くなりうるので 7s 遡って走査する。
            hi = bisect.bisect_left(_s_starts, end)
            lo = bisect.bisect_left(_s_starts, max(0.0, start - 7.0))
            for i in range(lo, hi):
                if trans[i]["end"] > start + 0.02:
                    return True
            return False

    candidates: list[tuple[float, float]] = []

    # 1e-9 の許容: 窓時刻は 0.02s の整数倍だが 0.02 は二進で厳密に表せず、整数倍
    # すると誤差が積もる（例 110*0.02 - 50*0.02 = 1.2000000000000002）。max_duration
    # ちょうど 1.2s の島が誤って弾かれるのを防ぐ。
    _DUR_EPS = 1e-9
    for sp_start, sp_end, sil_before, sil_after in islands:
        duration = sp_end - sp_start
        if (
            duration < min_island_duration - _DUR_EPS
            or duration > max_duration + _DUR_EPS
        ):
            continue
        if overlaps_transcript(sp_start, sp_end):
            continue
        # 両側の孤立条件。長い方が主しきい値、短い方が下限を満たすこと。
        sil_large = max(sil_before, sil_after)
        sil_small = min(sil_before, sil_after)
        if sil_large < min_adjacent_large or sil_small < min_adjacent_small:
            continue
        candidates.append((sp_start, sp_end))

    return candidates


def _preserve_unrecognized_gaps(
    segments: list[dict],
    cuts: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    source_duration: float,
    known_silences: list[tuple[float, float]] | None = None,
    uts_spans: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """主認識に無い非無音区間を未認識行として保持する（純関数）。

    speech は最終JSONと同じ元音源時刻へ変換した単語範囲、unrecognized は
    source_start/source_end を被覆範囲とする。中間gapは SILENCE_MIN_S より長い
    場合、先頭・末尾gapは長さを問わず、検出済み無音で完全被覆されない限り
    全体を unrecognized として追加する。部分的な無音被覆は安全側で全体保持する。

    uts_spans（detect_untranscribed_speech の出力）を渡すと、gap 内に入る UTS 島
    を `reason: "uts"` 付きの独立した unrecognized セグメントとして挿入する。挿入は
    通常gapの長さ判定（SILENCE_MIN_S 以下は行にしない）より前に行うため、短いgapに
    埋もれた未転写発話も候補行になる。長いgapは UTS 島と残余に分割し、残余は従来
    どおり reason なしの unrecognized として保持する（除外するのは検出済み無音で
    完全に説明できる部分のみ）。uts_spans が None/空なら出力は従来と完全に一致する。
    """
    duration = max(0.0, float(source_duration))
    known = silences if known_silences is None else known_silences
    # UTS島は使う前にソート・マージし、音源範囲 [0, duration] へクランプする。
    uts = _merge_time_spans(
        [
            (
                min(max(float(start), 0.0), duration),
                min(max(float(end), 0.0), duration),
            )
            for start, end in (uts_spans or [])
        ]
    )
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

    def _add_residual(start: float, end: float) -> None:
        """gap の残余（UTS島以外）を従来どおりの未認識行として追加する。"""
        if end <= start or _span_fully_covered(start, end, known):
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

    def _add_gap(start: float, end: float, edge: bool) -> None:
        if end <= start:
            return
        # UTS島の挿入は通常gapの判定より前に行う（短いgap内の候補も行にする）。
        islands = [
            (max(start, island_start), min(end, island_end))
            for island_start, island_end in uts
            if min(end, island_end) > max(start, island_start)
        ]
        for island_start, island_end in islands:
            added.append(
                {
                    "kind": "unrecognized",
                    "reason": "uts",
                    "source_start": island_start,
                    "source_end": island_end,
                    "text": "",
                    "words": [],
                }
            )
        # 残余の扱いは従来どおり: 検出済み無音で完全被覆されるgap、および
        # SILENCE_MIN_S 以下の中間gapは行にしない（判定は分割前のgap全体で行う）。
        if _span_fully_covered(start, end, known):
            return
        if not edge and end - start <= SILENCE_MIN_S:
            return
        cursor = start
        for island_start, island_end in islands:
            _add_residual(cursor, island_start)
            cursor = max(cursor, island_end)
        _add_residual(cursor, end)

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
    f. 短チャンク幻覚（時間密度異常・語数不問）: 文字数が HALLUC_MIN_CHARS 以上
       あり、発話時間（先頭単語 start 〜 末尾単語 end）が正で文字密度が
       HALLUC_DENSITY_CHARS_S 超、または発話時間が 0 以下なら除去。短チャンク
       turbo がチャンク境界でまれに出す定型幻覚（例「ご視聴ありがとうござい
       ました」14文字が0.12秒＝約116文字/秒）を捕まえる。条件 d と違い語数は
       問わない（境界幻覚は少語数で出るため）。除去分は既存のレスキュー機構に
       乗る。

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
        # 条件 f: 短チャンク幻覚（語数不問・文字数ベースの密度異常）
        if not drop[i] and len("".join(tokens)) >= HALLUC_MIN_CHARS:
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


def _detect_uts_spans(
    wav_path: str,
    segments: list[dict],
    cuts: list[tuple[float, float]],
    silences: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """16kHz wav と確定済み単語時刻から (UTS候補区間, 細かいRMS無音) を返す。

    wav_path は切り詰め前の 16kHz wav（元音源と同じ時間軸）。単語は
    _finalize_segment_words で元音源時刻へ変換したものを使い、時間軸を揃える。

    RMS無音（detect_rms_silences の結果）は UTS 検出の入力そのもので、render 側の
    guarded cut（未認識/UTS 削除時に元位置へ残すポーズの計算）でも必要になる。
    同じ音声を二度解析して値がずれるのを避けるため、ここで計算した結果を
    そのまま返して .sc.json へ保存する（呼び出し側で再計算しない）。

    音声を読めない場合は候補なし（空リスト）として扱う。UTS は候補の可視化だけを
    行う補助機能であり、ここで文字起こし全体を失敗させる理由がないため。
    """
    import soundfile as sf

    try:
        audio, samplerate = sf.read(wav_path, dtype="float32", always_2d=False)
    except (RuntimeError, OSError):
        return [], []
    if samplerate <= 0 or len(audio) == 0:
        return [], []

    words: list[dict] = []
    for seg in segments:
        if seg.get("kind") == "unrecognized":
            continue
        for word in _finalize_segment_words(seg, cuts, silences):
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            words.append({"start": float(start), "end": float(end)})

    word_spans = [(word["start"], word["end"]) for word in words]
    rms_silences = detect_rms_silences(audio, samplerate, word_spans)
    uts_spans = detect_untranscribed_speech(
        len(audio) / float(samplerate),
        rms_silences,
        [],
        transcript_words=words,
    )
    return uts_spans, rms_silences


def _silence_covered_fraction(
    start: float,
    end: float,
    merged_silences: list[tuple[float, float]],
) -> float:
    """[start, end] のうち merged_silences（マージ済み・昇順）に沈む割合を返す。

    end <= start（長さ0の単語）は区間ではなく点として扱い、その点が無音に
    含まれるなら 1.0、そうでなければ 0.0 を返す。
    """
    if end <= start:
        return 1.0 if any(s <= start <= e for s, e in merged_silences) else 0.0
    covered = 0.0
    for s, e in merged_silences:
        if e <= start:
            continue
        if s >= end:
            break
        covered += min(e, end) - max(s, start)
    return covered / (end - start)


def reattach_misplaced_words(
    words: list[dict],
    uts_spans: list[tuple[float, float]],
    rms_silences: list[tuple[float, float]],
    max_gap: float = UTS_REATTACH_MAX_GAP_S,
    min_silent_frac: float = UTS_REATTACH_MIN_SILENT_FRAC,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """UTS 島の隣接語がタイムスタンプを誤って無音側に持つ場合、島へ付け替える（純関数）。

    words は元音源時刻（start/end）で開始昇順・非重複とする（_finalize_segment_words
    の出力をそのまま結合したもの等）。各 dict の他のキーはそのまま保持する。

    各 UTS 島 (u_start, u_end) について:
    1. 「次の単語」= start >= u_end を満たす最初の未使用語。
       w.start - u_end <= max_gap かつ [w.start, w.end] の無音被覆率が
       min_silent_frac 以上、かつ付け替え後も順序が保てる
       （words 中でその直前の語の end <= u_start）なら、その語を
       [u_start, u_end] へ retime し、島を UTS 候補から外す。
    2. 1で付け替えられなければ「前の単語」= end <= u_start を満たす最後の
       未使用語について対称の条件（u_start - w.end <= max_gap、無音被覆、
       words 中でその直後の語の start >= u_end）を試す。
    3. いずれも成立しなければ島は UTS 候補として残す。

    1単語は最大1回しか付け替えない（他の島に使い回さない）。start/end が
    None の語は候補にしない。長さ0の語は、その1点が無音に含まれるときだけ
    「無音に沈んでいる」とみなす。

    戻り値: (付け替え反映後の words のコピー, 付け替えで消えなかった UTS 島)。
    入力の words/uts_spans/rms_silences は変更しない。
    """
    merged_silences = _merge_time_spans(rms_silences)
    out = [dict(w) for w in words]
    n = len(out)
    used = [False] * n
    remaining: list[tuple[float, float]] = []

    for u_start, u_end in uts_spans:
        u_start = float(u_start)
        u_end = float(u_end)
        reattached = False

        # 1. 次の単語を試す。
        next_idx = None
        for i in range(n):
            if used[i]:
                continue
            w_start = out[i].get("start")
            if w_start is None:
                continue
            if float(w_start) >= u_end:
                next_idx = i
                break
        if next_idx is not None:
            w_start = out[next_idx].get("start")
            w_end = out[next_idx].get("end")
            if w_start is not None and w_end is not None:
                w_start = float(w_start)
                w_end = float(w_end)
                prev_end = None
                if next_idx > 0:
                    prev_end = out[next_idx - 1].get("end")
                ordering_ok = prev_end is None or float(prev_end) <= u_start
                if (
                    w_start - u_end <= max_gap
                    and ordering_ok
                    and _silence_covered_fraction(
                        w_start, w_end, merged_silences
                    )
                    >= min_silent_frac
                ):
                    out[next_idx]["start"] = u_start
                    out[next_idx]["end"] = u_end
                    used[next_idx] = True
                    reattached = True

        # 2. 前の単語を試す。
        if not reattached:
            prev_idx = None
            for i in range(n - 1, -1, -1):
                if used[i]:
                    continue
                w_end = out[i].get("end")
                if w_end is None:
                    continue
                if float(w_end) <= u_start:
                    prev_idx = i
                    break
            if prev_idx is not None:
                w_start = out[prev_idx].get("start")
                w_end = out[prev_idx].get("end")
                if w_start is not None and w_end is not None:
                    w_start = float(w_start)
                    w_end = float(w_end)
                    next_start = None
                    if prev_idx + 1 < n:
                        next_start = out[prev_idx + 1].get("start")
                    ordering_ok = (
                        next_start is None or float(next_start) >= u_end
                    )
                    if (
                        u_start - w_end <= max_gap
                        and ordering_ok
                        and _silence_covered_fraction(
                            w_start, w_end, merged_silences
                        )
                        >= min_silent_frac
                    ):
                        out[prev_idx]["start"] = u_start
                        out[prev_idx]["end"] = u_end
                        used[prev_idx] = True
                        reattached = True

        if not reattached:
            remaining.append((u_start, u_end))

    return out, remaining


def _reattach_uts_words(
    segments: list[dict],
    cuts: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    uts_spans: list[tuple[float, float]],
    rms_silences: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """UTS 候補へ誤タイムの単語を付け替え、吸収できた候補を取り除く（結線用）。

    segments（kind!="unrecognized" のもの）の確定済み単語をまとめて
    reattach_misplaced_words へ渡し、付け替え結果を各 segment へ
    "_reattached_words" として書き込む（segments を直接書き換える）。以後の
    _finalize_segment_words 呼び出し（gap保持・ブロック割当・最終JSON書き出し）
    はこのキャッシュを読むため、補正が全呼び出し元へ自動的に伝播する。

    戻り値: 付け替えで消えなかった UTS 候補（_preserve_unrecognized_gaps へ渡す）。
    """
    flat_words: list[dict] = []
    owners: list[dict] = []
    for seg in segments:
        if seg.get("kind") == "unrecognized":
            continue
        for word in _finalize_segment_words(seg, cuts, silences):
            flat_words.append(word)
            owners.append(seg)

    corrected, remaining = reattach_misplaced_words(
        flat_words, uts_spans, rms_silences
    )

    by_seg: dict[int, list[dict]] = {}
    for seg, word in zip(owners, corrected):
        by_seg.setdefault(id(seg), []).append(word)
    for seg in segments:
        words = by_seg.get(id(seg))
        if words is not None:
            seg["_reattached_words"] = words

    return remaining


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
    auto_uts: bool = True,
) -> dict:
    """入力wavを文字起こしし、.sc.json / .sc.txt を生成する。

    model が None（未指定）のときは verbatim に応じて既定モデルを選ぶ:
    verbatim=True なら VERBATIM_MODEL、そうでなければ DEFAULT_MODEL。
    明示指定されたモデルはそのまま尊重する。

    verbatim=True のときはフィラーを転写するため各 30s チャンクに
    initial_prompt を渡す。condition_on_previous_text は False（短チャンク方式で
    繰り返し幻覚を断つ。非verbatim時もプロンプトなし・False）。

    auto_uts=True（既定）のときは UTS（未転写発話）候補行・plain 未認識区間行の
    どちらも〔〕付きで出力し、render 側でフィラーと同じ精密カット経路により
    既定でカットされる（行の〔〕を外せば保持、行削除でも従来どおりカット
    される）。auto_uts=False では旧来の「⚠ 未転写発話候補 N.N秒（音声保持）」
    「⚠ 未認識区間 N.N秒（音声保持）」表示に戻り、音声は常に保持される（render
    側は 〔〕 の有無だけを見るため、この引数は txt 出力の文言のみに影響し
    .sc.json の内容には影響しない）。

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
    # verbatim 本認識（2026-07-25 A/B 実験の結論に基づく turbo＋短チャンク方式）:
    # - モデルは turbo（VERBATIM_MODEL）。30s チャンクごとに initial_prompt
    #   （フィラー誘導文）を再投入し、全窓でフィラー捕捉を効かせる。
    # - condition_on_previous_text=False: 直前文脈を引き継がないことで、繰り返し
    #   幻覚の連鎖を断つ。短チャンク＝1窓なので文脈引き継ぎの利点は小さい。
    # - 温度フォールバックと hallucination_silence_threshold は無効にする（A/B
    #   実測でフィラー候補 67→3〜4 に激減する破壊的相互作用があった）:
    #   温度フォールバックは結果温度>0.5 でプロンプトをリセットして initial_prompt
    #   を消す。hallucination_silence_threshold は whisper の単語異常
    #   ヒューリスティクスがフィラーそのものを幻覚と誤判定して弾く。
    # verbatim の幻覚対策は後処理（drop_hallucinations 条件a〜f＋レスキュー）に
    # 委ねる。非verbatim とレスキュー再認識はフィラーを転写しないため両方有効。
    whisper_kwargs = dict(
        path_or_hf_repo=model,
        word_timestamps=True,
        language=lang,
        temperature=0 if verbatim else TEMPERATURE_FALLBACK,
        condition_on_previous_text=False,
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
    uts_spans: list[tuple[float, float]] = []
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
            # verbatim のレスキュー窓は large（RESCUE_MODEL）で丁寧に取り直す。
            # 非verbatim は本認識と同じモデルのまま。
            rescue_model = RESCUE_MODEL if verbatim else model
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
                        recog_wav, win[0], win[1], rescue_model, lang, True
                    )
                    if not rescued_segs:
                        attempts.append("safe")
                        rescued_segs = _rescue_transcribe(
                            recog_wav, win[0], win[1], rescue_model, lang, False
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

        # UTS（未転写発話）検出は 16kHz 一時wav が unlink される前に行う。
        # --verbatim の有無に関わらず常時実行する（verbatim は「フィラーも転写
        # する」モードであり、転写に載らない発声の検出とは別問題のため）。
        uts_spans, rms_silences = _detect_uts_spans(
            tmp_wav, kept_segments, cuts, periods
        )
        # Whisper が UTS 島の実発声を隣の無音側へ誤タイムした場合、その単語を
        # 島へ付け替えて UTS 候補から外す（単語自体を独立制御できるようにする）。
        # kept_segments を直接書き換え、以後の _finalize_segment_words 呼び出し
        # すべてへ補正が伝播する。
        uts_spans = _reattach_uts_words(
            kept_segments, cuts, periods, uts_spans, rms_silences
        )
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
        uts_spans=uts_spans,
    )

    segments_out = []
    lines: list[str] = []
    total_fillers = 0
    uts_count = 0

    for i, seg in enumerate(kept_segments):
        seg_id = _seg_id(i)
        if seg.get("kind") == "unrecognized":
            source_start = float(seg["source_start"])
            source_end = float(seg["source_end"])
            reason = seg.get("reason")
            out_seg = {
                "id": seg_id,
                "kind": "unrecognized",
                "source_start": source_start,
                "source_end": source_end,
                "text": seg.get("text", ""),
                "words": [],
            }
            # reason は UTS 候補のときだけ付ける。理由のない未認識行は従来どおりの
            # キー構成のままにし、旧 json の読み込み互換も壊さない。
            if reason:
                out_seg["reason"] = reason
            segments_out.append(out_seg)
            duration = max(0.0, source_end - source_start)
            if reason == "uts":
                uts_count += 1
                if auto_uts:
                    # 既定: フィラー提案と同じ「〔〕はカット」規約。
                    # 〔〕を外せば保持、行削除でも従来どおりカットされる。
                    lines.append(
                        f"[{seg_id} {_format_time(source_start)}] "
                        f"{FILLER_OPEN}未転写発話 {duration:.1f}秒{FILLER_CLOSE}"
                    )
                else:
                    lines.append(
                        f"[{seg_id} {_format_time(source_start)}] "
                        f"⚠ 未転写発話候補 {duration:.1f}秒（音声保持）"
                    )
            elif auto_uts:
                # 既定: UTS と同じ「〔〕はカット」規約。
                # 〔〕を外せば保持、行削除でも従来どおりカットされる。
                lines.append(
                    f"[{seg_id} {_format_time(source_start)}] "
                    f"{FILLER_OPEN}未認識 {duration:.1f}秒{FILLER_CLOSE}"
                )
            else:
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
        # UTS 検出で使った細かいRMS無音。render の guarded cut 専用
        # （粗い silences ではフィラー前後の短いポーズを拾えないため）。
        "rms_silences": [
            [round(s, 3), round(e, 3)] for s, e in rms_silences
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
        "uts_count": uts_count,
        "data": data,
    }


def _strip_wav_suffix(path: Path) -> Path:
    """拡張子(.wav等)を除いたベースパスを返す。"""
    return path.with_suffix("")
