"""音声の文字起こし: mlx-whisper で認識し .sc.json / .sc.txt を生成する。"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from . import fillers as fillers_mod

# 低信頼マーカー ◆ を付与する probability の閾値（これ未満で付与）
LOW_CONF_THRESHOLD = 0.5

# 認識用wavの無音切り詰め設定（元wavは不変。Whisper幻覚対策）
# silencedetect の noise 閾値(dB)・最小無音長(秒)、および切り詰め後に残す長さ(秒)
SILENCE_DB = -50  # 発話レベルが低い音源で静かな発話を無音と誤検出しないよう緩めに設定（認識用wav切り詰めのみに使用、render出力には影響しない）
SILENCE_MIN_S = 1.5
SILENCE_KEEP_S = 0.7

_SILENCE_RE = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")

# 表示用マーカー（render 側で除去する前提）
LOW_CONF_MARK = "◆"
FILLER_OPEN = "〔"
FILLER_CLOSE = "〕"

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


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


def _detect_silence(wav_path: str) -> str:
    """ffmpeg silencedetect を実行し stderr を返す。失敗時は空文字列。"""
    cmd = [
        "ffmpeg",
        "-i",
        wav_path,
        "-af",
        f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}",
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
                       filler_suggest: bool) -> tuple[str, int]:
    """1セグメントの .sc.txt 行文字列とフィラー候補数を組み立てる。

    表示テキストは json の words から再構成する。◆ と 〔〕 を取り除くと
    words の word 文字列の連結に一致する（render の文字diff→単語逆引き用）。
    """
    parts: list[str] = []
    filler_count = 0
    for w in words:
        raw = w.get("word", "")
        core = raw.lstrip()
        lead = raw[: len(raw) - len(core)]
        prob = w.get("probability")

        piece = core
        if filler_suggest and fillers_mod.is_filler(raw, fillers):
            piece = f"{FILLER_OPEN}{piece}{FILLER_CLOSE}"
            filler_count += 1
        if prob is not None and prob < LOW_CONF_THRESHOLD:
            piece = f"{LOW_CONF_MARK}{piece}"
        parts.append(lead + piece)

    text = "".join(parts)
    if words:
        start = words[0].get("start")
        if start is not None:
            return f"[{seg_id} {_format_time(start)}] {text}", filler_count
    return f"[{seg_id}] {text}", filler_count


def transcribe(
    input_wav: str,
    lang: str | None = "ja",
    model: str = DEFAULT_MODEL,
    filler_suggest: bool = True,
) -> dict:
    """入力wavを文字起こしし、.sc.json / .sc.txt を生成する。

    戻り値: {"json_path", "txt_path", "filler_count", "data"}
    """
    import mlx_whisper

    input_path = Path(input_wav)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_wav}")

    tmp_wav = _convert_to_16k_mono(str(input_path))
    # 認識用wavのみ長い無音を切り詰める（元wavは不変）。単語時刻は後段で元時刻へ逆写像。
    cuts = build_cuts(parse_silence_periods(_detect_silence(tmp_wav)))
    trimmed_wav, removed_s = _write_trimmed_wav(tmp_wav, cuts)
    recog_wav = trimmed_wav
    try:
        result = mlx_whisper.transcribe(
            recog_wav,
            path_or_hf_repo=model,
            word_timestamps=True,
            language=lang,
            temperature=0,
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            compression_ratio_threshold=2.0,
        )
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

    for i, seg in enumerate(result.get("segments", [])):
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
        segments_out.append(
            {
                "id": seg_id,
                "text": seg.get("text", ""),
                "words": words,
            }
        )
        line, fcount = build_segment_line(seg_id, words, fillers, filler_suggest)
        lines.append(line)
        total_fillers += fcount

    data = {
        "source_wav": str(input_path.resolve()),
        "language": result.get("language", lang),
        "model": model,
        "silence_trim": {
            "count": len(cuts),
            "removed_s": round(removed_s, 3),
        },
        "segments": segments_out,
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
        "data": data,
    }


def _strip_wav_suffix(path: Path) -> Path:
    """拡張子(.wav等)を除いたベースパスを返す。"""
    return path.with_suffix("")
