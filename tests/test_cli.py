"""stefnceorf.cli の単体テスト。"""

from __future__ import annotations

import sys
import types

import pytest

from stefnceorf import cli
from stefnceorf import transcribe as transcribe_mod


# ---- _build_parser ----

def test_parser_trans_alias_command_name():
    """'trans' エイリアスで parse すると args.command == 'trans' になる。"""
    args = cli._build_parser().parse_args(["trans", "x.wav"])
    assert args.command == "trans"


def test_parser_trans_alias_input():
    """'trans' エイリアスで parse すると args.input が正しく設定される。"""
    args = cli._build_parser().parse_args(["trans", "x.wav"])
    assert args.input == "x.wav"


def test_parser_trans_alias_defaults():
    """'trans' エイリアスで parse したとき各オプションが既定値になる。"""
    args = cli._build_parser().parse_args(["trans", "x.wav"])
    assert args.lang == "ja"
    assert args.model is None
    assert args.filler_suggest is False
    assert args.verbatim is False
    assert args.pause_threshold == pytest.approx(transcribe_mod.PAUSE_THRESHOLD_S)


def test_parser_transcribe_still_works():
    """'transcribe' の元コマンドが引き続き機能する。"""
    args = cli._build_parser().parse_args(["transcribe", "y.wav"])
    assert args.command == "transcribe"
    assert args.input == "y.wav"


# ---- main() のディスパッチ ----

@pytest.fixture
def fake_transcribe(monkeypatch, tmp_path):
    """stefnceorf.transcribe.transcribe を差し替え、呼び出し引数を記録する。"""
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"dummy")

    calls = []

    def _fake(input_wav, **kwargs):
        calls.append({"input_wav": input_wav, "kwargs": kwargs})
        return {
            "txt_path": str(tmp_path / "x.sc.txt"),
            "json_path": str(tmp_path / "x.sc.json"),
            "filler_count": 0,
            "silence_cut_count": 0,
            "silence_removed_s": 0.0,
            "hallucination_drop_count": 0,
            "hallucination_ranges": [],
        }

    monkeypatch.setattr("stefnceorf.transcribe.transcribe", _fake)
    return calls, wav


def test_main_trans_calls_transcribe(fake_transcribe, tmp_path):
    """main(['trans', 'x.wav']) で transcribe 関数が呼ばれる。"""
    calls, wav = fake_transcribe
    ret = cli.main(["trans", str(wav)])
    assert ret == 0
    assert len(calls) == 1
    assert calls[0]["input_wav"] == str(wav)


def test_main_trans_passes_lang(fake_transcribe, tmp_path):
    """main(['trans', ..., '--lang', 'en']) で lang が渡される。"""
    calls, wav = fake_transcribe
    cli.main(["trans", str(wav), "--lang", "en"])
    assert calls[0]["kwargs"]["lang"] == "en"
