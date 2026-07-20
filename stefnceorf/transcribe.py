"""音声の文字起こし: mlx-whisper で認識し .sc.json / .sc.txt を生成する。"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from . import fillers as fillers_mod

# 低信頼マーカー ◆ を付与する probability の閾値（これ未満で付与）
LOW_CONF_THRESHOLD = 0.5

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
    try:
        result = mlx_whisper.transcribe(
            tmp_wav,
            path_or_hf_repo=model,
            word_timestamps=True,
            language=lang,
            temperature=0,
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            compression_ratio_threshold=2.0,
        )
    finally:
        try:
            os.unlink(tmp_wav)
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
        "data": data,
    }


def _strip_wav_suffix(path: Path) -> Path:
    """拡張子(.wav等)を除いたベースパスを返す。"""
    return path.with_suffix("")
