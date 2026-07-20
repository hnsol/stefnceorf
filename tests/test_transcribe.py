import json
import sys
import types

import pytest

from stefnceorf import transcribe
from stefnceorf import fillers as fillers_mod


# ---- build_segment_line 単体テスト（純関数） ----

def _words(*items):
    """(word, prob) のタプル列から words リストを作る。"""
    out = []
    for w, p in items:
        out.append({"word": w, "start": 0.0, "end": 1.0, "probability": p})
    return out


def test_line_plain():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("結局", 0.9), ("面倒", 0.9))
    line, fc = transcribe.build_segment_line("0001", words, ja, True)
    assert line == "[0001] 結局面倒"
    assert fc == 0


def test_line_low_conf_mark():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("結局", 0.9), ("面倒", 0.3))
    line, fc = transcribe.build_segment_line("0001", words, ja, True)
    assert line == "[0001] 結局◆面倒"
    assert fc == 0


def test_line_filler_wrapped():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("とか", 0.9))
    line, fc = transcribe.build_segment_line("0002", words, ja, True)
    assert line == "[0002] 〔まあ〕とか"
    assert fc == 1


def test_line_no_filler_suggest():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("とか", 0.9))
    line, fc = transcribe.build_segment_line("0002", words, ja, False)
    assert line == "[0002] まあとか"
    assert fc == 0


def test_line_marker_removal_matches_word_concat():
    """◆と〔〕を除くと words の連結に一致する（render の文字対応要件）。"""
    ja = fillers_mod.load_fillers("ja")
    words = _words(("あのー", 0.9), ("動画", 0.3), ("です", 0.9))
    line, _ = transcribe.build_segment_line("0003", words, ja, True)
    body = line.split("] ", 1)[1]
    cleaned = (
        body.replace(transcribe.LOW_CONF_MARK, "")
        .replace(transcribe.FILLER_OPEN, "")
        .replace(transcribe.FILLER_CLOSE, "")
    )
    assert cleaned == "".join(w["word"] for w in words)


def test_line_english_preserves_spaces():
    en = fillers_mod.load_fillers("en")
    words = _words((" hello", 0.9), (" um", 0.9), (" world", 0.3))
    line, fc = transcribe.build_segment_line("0001", words, en, True)
    body = line.split("] ", 1)[1]
    cleaned = (
        body.replace(transcribe.LOW_CONF_MARK, "")
        .replace(transcribe.FILLER_OPEN, "")
        .replace(transcribe.FILLER_CLOSE, "")
    )
    assert cleaned == " hello um world"
    assert fc == 1


# ---- transcribe 全体（mlx_whisper と ffmpeg をモック） ----

@pytest.fixture
def fake_whisper(monkeypatch):
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "結局まあ面倒",
                "words": [
                    {"word": "結局", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": "まあ", "start": 0.5, "end": 0.7, "probability": 0.9},
                    {"word": "面倒", "start": 0.7, "end": 1.2, "probability": 0.3},
                ],
            },
            {
                "text": "作業",
                "words": [
                    {"word": "作業", "start": 1.2, "end": 1.8, "probability": 0.9},
                ],
            },
        ],
    }
    captured = {}

    def fake_transcribe(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return fake_result

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    # ffmpeg変換は行わずダミーパスを返す
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    return captured


def _make_input(tmp_path):
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"RIFFdummy")
    return wav


def test_transcribe_generates_files(tmp_path, fake_whisper):
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")

    assert res["json_path"].endswith("input.sc.json")
    assert res["txt_path"].endswith("input.sc.txt")

    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))
    assert data["source_wav"].endswith("input.wav")
    assert data["model"] == transcribe.DEFAULT_MODEL
    assert len(data["segments"]) == 2
    assert data["segments"][0]["id"] == "0001"
    assert data["segments"][1]["id"] == "0002"
    assert data["segments"][0]["words"][0] == {
        "word": "結局",
        "start": 0.0,
        "end": 0.5,
        "probability": 0.95,
    }


def test_transcribe_txt_markers(tmp_path, fake_whisper):
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    # 〔まあ〕 と 面倒(prob0.3)の◆
    assert lines[0] == "[0001] 結局〔まあ〕◆面倒"
    assert lines[1] == "[0002] 作業"
    assert res["filler_count"] == 1


def test_transcribe_no_filler_suggest(tmp_path, fake_whisper):
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", filler_suggest=False)
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    assert lines[0] == "[0001] 結局まあ◆面倒"
    assert res["filler_count"] == 0


def test_transcribe_passes_whisper_kwargs(tmp_path, fake_whisper):
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav), lang="ja", model="custom-model")
    kw = fake_whisper["kwargs"]
    assert kw["path_or_hf_repo"] == "custom-model"
    assert kw["word_timestamps"] is True
    assert kw["language"] == "ja"
    assert kw["temperature"] == 0
    assert kw["condition_on_previous_text"] is False
    assert kw["no_speech_threshold"] == 0.8
    assert kw["compression_ratio_threshold"] == 2.0


def test_transcribe_missing_input(tmp_path, fake_whisper):
    with pytest.raises(FileNotFoundError):
        transcribe.transcribe(str(tmp_path / "nope.wav"))


@pytest.fixture
def fake_whisper_en(monkeypatch):
    """自動判定で英語になるケース（result['language'] == 'en'）。"""
    fake_result = {
        "language": "en",
        "segments": [
            {
                "text": " so um yeah",
                "words": [
                    {"word": " so", "start": 0.0, "end": 0.3, "probability": 0.9},
                    {"word": " um", "start": 0.3, "end": 0.5, "probability": 0.9},
                    {"word": " yeah", "start": 0.5, "end": 0.9, "probability": 0.9},
                ],
            },
        ],
    }

    def fake_transcribe(path, **kwargs):
        return fake_result

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    return fake_result


def test_transcribe_default_lang_is_ja(tmp_path, fake_whisper):
    """lang未指定時の既定は 'ja' であり、Whisperに language='ja' が渡される。"""
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav))
    assert fake_whisper["kwargs"]["language"] == "ja"


def test_transcribe_auto_lang_uses_detected_dict(tmp_path, fake_whisper_en):
    """--lang未指定でも result['language']=='en' なら en辞書で 'um' を検出する。"""
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang=None)
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    assert lines[0] == "[0001]  so 〔um〕 yeah"
    assert res["filler_count"] == 1
