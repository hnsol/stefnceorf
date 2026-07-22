"""stefnceorf.cli の単体テスト。"""

from __future__ import annotations

import sys
import types

import pytest

from stefnceorf import cli
from stefnceorf import transcribe as transcribe_mod
from stefnceorf.render import GAP_MAX_S, GAP_THRESHOLD_S


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
    assert args.filler_suggest is True
    assert args.verbatim is True
    assert args.pause_threshold == pytest.approx(transcribe_mod.PAUSE_THRESHOLD_S)


def test_parser_verbatim_filler_default_on():
    """引数なしで verbatim / filler_suggest が既定 True になる。"""
    args = cli._build_parser().parse_args(["transcribe", "x.wav"])
    assert args.verbatim is True
    assert args.filler_suggest is True


def test_parser_no_verbatim_no_filler_disable():
    """--no-verbatim / --no-filler-suggest で False になる。"""
    args = cli._build_parser().parse_args(
        ["transcribe", "x.wav", "--no-verbatim", "--no-filler-suggest"]
    )
    assert args.verbatim is False
    assert args.filler_suggest is False


def test_parser_transcribe_still_works():
    """'transcribe' の元コマンドが引き続き機能する。"""
    args = cli._build_parser().parse_args(["transcribe", "y.wav"])
    assert args.command == "transcribe"
    assert args.input == "y.wav"


def test_parser_logic_defaults():
    args = cli._build_parser().parse_args(["logic", "input.sc.txt"])
    assert args.command == "logic"
    assert args.input == "input.sc.txt"
    assert args.output is None
    assert args.gap_threshold == GAP_THRESHOLD_S
    assert args.gap_max == GAP_MAX_S


def test_parser_logic_options():
    args = cli._build_parser().parse_args(
        [
            "logic",
            "input.sc.txt",
            "-o",
            "output.fcpxml",
            "--gap-threshold",
            "2.5",
            "--gap-max",
            "0.75",
        ]
    )
    assert args.output == "output.fcpxml"
    assert args.gap_threshold == pytest.approx(2.5)
    assert args.gap_max == pytest.approx(0.75)


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


def test_main_logic_dispatches_and_prints_generated(monkeypatch, capsys):
    calls = []

    def _fake(input_path, **kwargs):
        calls.append((input_path, kwargs))
        return "custom.fcpxml"

    monkeypatch.setattr("stefnceorf.fcpxml.export_fcpxml", _fake)
    rc = cli.main(
        [
            "logic",
            "input.sc.txt",
            "-o",
            "custom.fcpxml",
            "--gap-threshold",
            "2.0",
            "--gap-max",
            "0.5",
        ]
    )

    assert rc == 0
    assert calls == [
        (
            "input.sc.txt",
            {"output": "custom.fcpxml", "gap_threshold": 2.0, "gap_max": 0.5},
        )
    ]
    assert capsys.readouterr().out == "生成: custom.fcpxml\n"


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        RuntimeError("runtime"),
        ValueError("bad value"),
        OSError("cannot write"),
    ],
)
def test_main_logic_expected_errors_return_one(monkeypatch, capsys, error):
    def _fake(*args, **kwargs):
        raise error

    monkeypatch.setattr("stefnceorf.fcpxml.export_fcpxml", _fake)
    assert cli.main(["logic", "input.sc.txt"]) == 1
    assert capsys.readouterr().err == f"エラー: {error}\n"
