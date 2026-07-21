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

# ポーズベース区切り（ブロック単位削除）設定
# この秒数以上の無音を「カット可能なポーズ区切り」としてブロック境界にする
PAUSE_THRESHOLD_S = 0.15

# 繰り返し幻覚セグメントの後処理除去（--verbatim の condition_on_previous_text=True
# で長尺の静音区間から発生する繰り返し幻覚への対策。入力側では完全に防げない）
HALLUC_MIN_WORDS = 5     # セグメント内反復判定の最小語数
HALLUC_TOP_RATIO = 0.7   # 最頻トークンの占有率閾値
HALLUC_RUN = 3           # 同一テキストセグメントの連続数閾値

# 幻覚区間の再認識レスキュー窓の最小長（秒）。これ未満の窓は内容なしとみなし
# レスキューを省略する（従来通り除去扱い）。
RESCUE_MIN_WINDOW_S = 0.5

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

    戻り値: (残すセグメント列, 除去したセグメント列)。順序は入力順を保つ。
    """
    n = len(segments)
    drop = [False] * n

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
    start = 0.0 if prev_kept_end is None else prev_kept_end
    end = wav_duration if next_kept_start is None else next_kept_start
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


def _rescue_kwargs(model: str, lang: str | None) -> dict:
    """レスキュー再認識用の安全設定 whisper kwargs（プロンプトなし・condT False）。"""
    return dict(
        path_or_hf_repo=model,
        word_timestamps=True,
        language=lang,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.8,
        compression_ratio_threshold=2.0,
    )


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
    recog_wav: str,
    win_start: float,
    win_end: float,
    model: str,
    lang: str | None,
) -> list[dict]:
    """レスキュー窓を安全設定で再認識し、幻覚ガード後の生存セグメント列を返す。

    単語時刻には窓開始時刻を加算し認識用wav時刻へ戻す。得られたセグメントにも
    drop_hallucinations を1回だけ適用する（再帰レスキューはしない）。
    """
    import mlx_whisper

    clip = _slice_wav(recog_wav, win_start, win_end)
    try:
        result = mlx_whisper.transcribe(clip, **_rescue_kwargs(model, lang))
    finally:
        try:
            os.unlink(clip)
        except OSError:
            pass

    rsegs = result.get("segments", []) or []
    # 窓内の相対時刻を認識用wav時刻へ戻す（窓開始を加算）
    for seg in rsegs:
        for w in seg.get("words") or []:
            if w.get("start") is not None:
                w["start"] = float(w["start"]) + win_start
            if w.get("end") is not None:
                w["end"] = float(w["end"]) + win_start
    kept, _dropped = drop_hallucinations(rsegs)
    return kept


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


def build_segment_line(seg_id: str, words: list[dict], fillers: set[str],
                       filler_suggest: bool,
                       blocks: list[int] | None = None) -> tuple[str, int]:
    """1セグメントの .sc.txt 行文字列とフィラー候補数を組み立てる。

    表示テキストは json の words から再構成する。◆・〔〕・／ を取り除くと
    words の word 文字列の連結に一致する（render の文字diff→単語逆引き用）。

    blocks が渡された場合、ブロック境界（blocks[i] != blocks[i-1]）の直前に
    区切り記号 ／ を挿入する（lead 空白・〔〕の外側、◆の前）。
    """
    parts: list[str] = []
    filler_count = 0
    for idx, w in enumerate(words):
        raw = w.get("word", "")
        core = raw.lstrip()
        lead = raw[: len(raw) - len(core)]
        prob = w.get("probability")

        piece = core
        if filler_suggest and fillers_mod.is_filler(raw, fillers):
            # 単独ブロック（前後にポーズ）のフィラーだけ〔〕提案する。
            # 他の語と同ブロックのフィラーは消すと巻き込みが起きるため提案しない。
            is_solo = True
            if blocks is not None:
                b = blocks[idx]
                is_solo = not any(
                    blocks[j] == b for j in range(len(blocks)) if j != idx
                )
            if is_solo:
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
    cuts = build_cuts(cut_periods)
    trimmed_wav, removed_s = _write_trimmed_wav(tmp_wav, cuts)
    recog_wav = trimmed_wav
    whisper_kwargs = dict(
        path_or_hf_repo=model,
        word_timestamps=True,
        language=lang,
        temperature=0,
        condition_on_previous_text=verbatim,
        no_speech_threshold=0.8,
        compression_ratio_threshold=2.0,
        verbose=False,  # mlx-whisper 組み込みの進捗バー(tqdm)を有効化
    )
    if verbatim:
        whisper_kwargs["initial_prompt"] = (
            FILLER_PROMPT_EN if lang == "en" else FILLER_PROMPT
        )
    # 幻覚除去グループを再認識レスキューする間、認識用wav（trimmed_wav）を
    # 生存させる必要があるため、本認識からレスキュー完了までを1つの try で囲う。
    halluc_ranges: list[dict] = []
    dropped_count = 0
    try:
        result = mlx_whisper.transcribe(recog_wav, **whisper_kwargs)

        # 繰り返し幻覚とみられるセグメントをループ前に除去する。
        raw_segments = result.get("segments", []) or []
        kept_segments, dropped_segments = drop_hallucinations(raw_segments)
        dropped_count = len(dropped_segments)

        # 除去グループごとに kept 境界からレスキュー窓を決め、安全設定で再認識して
        # 差し替える。連続する除去を1範囲にまとめ、元音源時刻で報告する。
        if dropped_segments:
            dropped_ids = {id(s) for s in dropped_segments}
            wav_dur = _wav_duration(recog_wav)
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
                # 除去単語時刻（元音源時刻）と先頭サンプルを収集
                starts: list[float] = []
                ends: list[float] = []
                sample = ""
                for gseg in group:
                    for w in gseg.get("words") or []:
                        if w.get("start") is not None:
                            starts.append(rec_to_src(float(w["start"]), cuts))
                        if w.get("end") is not None:
                            ends.append(rec_to_src(float(w["end"]), cuts))
                    if not sample:
                        sample = (gseg.get("text") or "").strip()[:20]

                prev_end = _seg_last_end(prev_kept) if prev_kept else None
                next_start = (
                    _seg_first_start(next_kept) if next_kept else None
                )
                win = rescue_window(prev_end, next_start, wav_dur)

                # 窓もテキストも時刻も無い純粋な空セグメント群は音声内容を
                # 失わないため報告もレスキューもしない（従来挙動）
                if win is None and not starts and not sample:
                    return []

                rescued_segs: list[dict] = []
                if win is not None:
                    rescued_segs = _rescue_transcribe(
                        recog_wav, win[0], win[1], model, lang
                    )

                # 報告時刻: 除去単語時刻を優先し、無ければ窓境界を逆写像する
                if starts:
                    rng_start: float | None = min(starts)
                elif win is not None:
                    rng_start = rec_to_src(win[0], cuts)
                else:
                    rng_start = None
                if ends:
                    rng_end: float | None = max(ends)
                elif win is not None:
                    rng_end = rec_to_src(win[1], cuts)
                else:
                    rng_end = None

                halluc_ranges.append(
                    {
                        "start": rng_start,
                        "end": rng_end,
                        "sample": sample,
                        "rescued": bool(rescued_segs),
                        "rescued_segments": len(rescued_segs),
                    }
                )
                return rescued_segs

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

    segments_out = []
    lines: list[str] = []
    total_fillers = 0

    for i, seg in enumerate(kept_segments):
        seg_id = _seg_id(i)
        words = []
        for w in seg.get("words", []) or []:
            words.append(
                {
                    "word": w.get("word", ""),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "probability": w.get("probability"),
                }
            )
        # 認識用wavの時刻を元音源の時刻へ逆写像（cuts が空なら恒等）
        words = remap_words(words, cuts)
        # ポーズベースのブロック割当（各 word に "block" を付与）
        blocks = assign_blocks(words, block_silences, pause_threshold)
        for wi, w in enumerate(words):
            w["block"] = blocks[wi]
        segments_out.append(
            {
                "id": seg_id,
                "text": seg.get("text", ""),
                "words": words,
            }
        )
        line, fcount = build_segment_line(
            seg_id, words, fillers, filler_suggest, blocks=blocks
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
