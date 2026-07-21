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
    assert line == "[0001 0:00] 結局面倒"
    assert fc == 0


def test_line_low_conf_mark():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("結局", 0.9), ("面倒", 0.3))
    line, fc = transcribe.build_segment_line("0001", words, ja, True)
    assert line == "[0001 0:00] 結局◆面倒"
    assert fc == 0


def test_line_filler_wrapped():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("とか", 0.9))
    line, fc = transcribe.build_segment_line("0002", words, ja, True)
    assert line == "[0002 0:00] 〔まあ〕とか"
    assert fc == 1


def test_line_no_filler_suggest():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("とか", 0.9))
    line, fc = transcribe.build_segment_line("0002", words, ja, False)
    assert line == "[0002 0:00] まあとか"
    assert fc == 0


def test_line_empty_words_no_timestamp():
    """words が空のときは時刻なし [ID] 形式になる。"""
    ja = fillers_mod.load_fillers("ja")
    line, fc = transcribe.build_segment_line("0001", [], ja, True)
    assert line == "[0001] "
    assert fc == 0


def test_line_timestamp_over_60min():
    """60分以上のセグメントは H:MM:SS 形式で表示される。"""
    ja = fillers_mod.load_fillers("ja")
    words = [{"word": "テスト", "start": 3723.9, "end": 3724.5, "probability": 0.9}]
    line, _ = transcribe.build_segment_line("0001", words, ja, False)
    # 3723.9 → floor → 3723秒 = 1時間2分3秒
    assert line == "[0001 1:02:03] テスト"


def test_format_time_various():
    """_format_time のフォーマット確認。"""
    assert transcribe._format_time(0.0) == "0:00"
    assert transcribe._format_time(5.9) == "0:05"
    assert transcribe._format_time(754.0) == "12:34"
    assert transcribe._format_time(3600.0) == "1:00:00"
    assert transcribe._format_time(3723.0) == "1:02:03"


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


# ---- 無音切り詰め: silencedetect パース / cuts / 逆写像（純関数） ----

def test_parse_silence_periods():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 12.345\n"
        "[silencedetect @ 0x1] silence_end: 30.678 | silence_duration: 18.333\n"
        "[silencedetect @ 0x1] silence_start: 40.0\n"
        "[silencedetect @ 0x1] silence_end: 42.5 | silence_duration: 2.5\n"
    )
    assert transcribe.parse_silence_periods(stderr) == [
        (12.345, 30.678),
        (40.0, 42.5),
    ]


def test_parse_silence_periods_unmatched_start_ignored():
    stderr = (
        "silence_start: 5.0\n"
        "silence_end: 8.0\n"
        "silence_start: 20.0\n"  # 末尾まで無音、end なし
    )
    assert transcribe.parse_silence_periods(stderr) == [(5.0, 8.0)]


def test_parse_silence_periods_empty():
    assert transcribe.parse_silence_periods("no silence here") == []


def test_build_cuts_filters_short_and_computes_middle():
    # 無音 [10,13] 長さ3.0 > keep0.7 → cut [10.35, 12.65] len 2.3
    # 無音 [20,20.5] 長さ0.5 ≤ keep → 除外
    cuts = transcribe.build_cuts([(10.0, 13.0), (20.0, 20.5)], keep=0.7)
    assert cuts == [(pytest.approx(10.35), pytest.approx(2.3))]


def test_rec_to_src_identity_no_cuts():
    for t in [0.0, 1.5, 100.0]:
        assert transcribe.rec_to_src(t, []) == t


def test_rec_to_src_single_cut():
    # cut at src 10.35, len 2.3 → rec点 10.35。以降の rec 時刻に +2.3
    cuts = [(10.35, 2.3)]
    assert transcribe.rec_to_src(5.0, cuts) == pytest.approx(5.0)  # 前
    assert transcribe.rec_to_src(10.35, cuts) == pytest.approx(10.35)  # 境界=前寄せ
    assert transcribe.rec_to_src(11.0, cuts) == pytest.approx(13.3)  # 後


def test_rec_to_src_multiple_cuts():
    # cut1: src10, len2 (rec点10) / cut2: src20, len3 (rec点 20-2=18)
    cuts = [(10.0, 2.0), (20.0, 3.0)]
    assert transcribe.rec_to_src(5.0, cuts) == pytest.approx(5.0)
    assert transcribe.rec_to_src(15.0, cuts) == pytest.approx(17.0)  # cut1後
    assert transcribe.rec_to_src(19.0, cuts) == pytest.approx(24.0)  # 両cut後


def test_rec_to_src_boundary_clamps_to_near():
    # カット点ちょうどはソース側の近傍境界（カット開始）へクランプ
    cuts = [(10.0, 5.0)]
    assert transcribe.rec_to_src(10.0, cuts) == pytest.approx(10.0)


def test_remap_words_identity():
    words = [{"word": "a", "start": 1.0, "end": 2.0, "probability": 0.9}]
    out = transcribe.remap_words(words, [])
    assert out[0]["start"] == 1.0 and out[0]["end"] == 2.0


def test_remap_words_applies_and_monotonic():
    cuts = [(10.0, 2.0)]
    words = [
        {"word": "a", "start": 5.0, "end": 9.0, "probability": 0.9},
        {"word": "b", "start": 11.0, "end": 12.0, "probability": 0.9},
    ]
    out = transcribe.remap_words(words, cuts)
    assert out[0]["start"] == pytest.approx(5.0)
    assert out[0]["end"] == pytest.approx(9.0)
    assert out[1]["start"] == pytest.approx(13.0)
    assert out[1]["end"] == pytest.approx(14.0)
    # 単調性
    assert out[0]["end"] <= out[1]["start"]


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
    # ffmpeg変換・無音検出は行わない（無音なし＝逆写像は恒等）
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p: "")
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
    res = transcribe.transcribe(str(wav), lang="ja", filler_suggest=True)
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    # seg1: 最初の単語 start=0.0 → 0:00、〔まあ〕 と 面倒(prob0.3)の◆
    assert lines[0] == "[0001 0:00] 結局〔まあ〕◆面倒"
    # seg2: 最初の単語 start=1.2 → floor=1秒 → 0:01
    assert lines[1] == "[0002 0:01] 作業"
    assert res["filler_count"] == 1


def test_transcribe_no_filler_suggest(tmp_path, fake_whisper):
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", filler_suggest=False)
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    assert lines[0] == "[0001 0:00] 結局まあ◆面倒"
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
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p: "")
    return fake_result


def test_transcribe_silence_trim_remaps_and_meta(tmp_path, monkeypatch):
    """無音検出あり: 認識用wavを切り詰め、単語時刻を元時刻へ逆写像しメタを付与する。"""
    import numpy as np
    import soundfile as sf

    sr = 16000
    wav = tmp_path / "input.wav"
    sf.write(str(wav), np.zeros(10 * sr, dtype="float32"), sr)

    # 無音 [2,5]（3s）→ cut (2.35, 2.3)。rec点 2.35 以降は +2.3。
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "前後",
                "words": [
                    {"word": "前", "start": 1.0, "end": 2.0, "probability": 0.9},
                    {"word": "後", "start": 3.0, "end": 4.0, "probability": 0.9},
                ],
            },
        ],
    }
    captured = {}

    def fake_transcribe(path, **kwargs):
        captured["path"] = path
        return fake_result

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: str(wav))
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p: "silence_start: 2.0\nsilence_end: 5.0 | silence_duration: 3.0\n",
    )

    res = transcribe.transcribe(str(wav), lang="ja")
    data = res["data"]
    w = data["segments"][0]["words"]
    # 「前」rec[1,2] は cut点2.35より前 → そのまま
    assert w[0]["start"] == pytest.approx(1.0)
    assert w[0]["end"] == pytest.approx(2.0)
    # 「後」rec[3,4] は cut点以降 → +2.3
    assert w[1]["start"] == pytest.approx(5.3)
    assert w[1]["end"] == pytest.approx(6.3)
    assert data["silence_trim"]["count"] == 1
    assert data["silence_trim"]["removed_s"] == pytest.approx(2.3, abs=0.01)
    assert res["silence_cut_count"] == 1


def test_transcribe_default_lang_is_ja(tmp_path, fake_whisper):
    """lang未指定時の既定は 'ja' であり、Whisperに language='ja' が渡される。"""
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav))
    assert fake_whisper["kwargs"]["language"] == "ja"


def test_transcribe_auto_lang_uses_detected_dict(tmp_path, fake_whisper_en):
    """--lang未指定でも result['language']=='en' なら en辞書で 'um' を検出する。"""
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang=None, filler_suggest=True)
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    assert lines[0] == "[0001 0:00]  so 〔um〕 yeah"
    assert res["filler_count"] == 1
