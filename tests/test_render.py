import json

import numpy as np
import pytest

from stefnceorf import render


# ---- parse_edited_txt ----

def test_parse_plain():
    txt = "[0001] 結局面倒\n[0002] 作業\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒"), ("0002", "作業")]


def test_parse_with_timestamp():
    """新形式 [ID 時刻] はIDのみ抽出し、テキストは正常にパースされる。"""
    txt = "[0001 0:12] 結局面倒\n[0002 12:34] 作業\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒"), ("0002", "作業")]


def test_parse_timestamp_hhmmss():
    """H:MM:SS 形式の時刻付き行もIDのみ抽出できる。"""
    txt = "[0001 1:02:03] テスト\n"
    assert render.parse_edited_txt(txt) == [("0001", "テスト")]


def test_parse_legacy_and_new_mixed():
    """旧形式と新形式が混在してもそれぞれ正しくパースされる。"""
    txt = "[0001] 旧形式\n[0002 0:05] 新形式\n"
    assert render.parse_edited_txt(txt) == [("0001", "旧形式"), ("0002", "新形式")]


def test_parse_ignores_blank_lines():
    txt = "[0001] あ\n\n   \n[0002] い\n"
    assert render.parse_edited_txt(txt) == [("0001", "あ"), ("0002", "い")]


def test_parse_strips_low_conf_mark():
    txt = "[0001] 結局◆面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒")]


def test_parse_filler_bracket_deleted():
    # 〔...〕 は中身ごと削除
    txt = "[0001] 結局〔まあ〕面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒")]


def test_parse_filler_bracket_removed_keeps_text():
    # 括弧を外した箇所は通常文字として残る
    txt = "[0001] 結局まあ面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局まあ面倒")]


def test_parse_broken_id_line_raises():
    with pytest.raises(ValueError):
        render.parse_edited_txt("結局面倒\n")


def test_parse_duplicate_id_raises():
    with pytest.raises(ValueError):
        render.parse_edited_txt("[0001] あ\n[0001] い\n")


# ---- surviving_words ----

def test_survive_no_edit():
    words = ["結局", "面倒", "作業"]
    keep, warns = render.surviving_words(words, "結局面倒作業")
    assert keep == {0, 1, 2}
    assert warns == []


def test_survive_word_deleted():
    words = ["結局", "面倒", "作業"]
    keep, warns = render.surviving_words(words, "結局作業")
    assert keep == {0, 2}
    assert warns == []


def test_survive_partial_char_delete_removes_word():
    # 単語の一部文字だけ削除 → その単語ごと削除
    words = ["結局", "面倒", "作業"]
    keep, warns = render.surviving_words(words, "結局倒作業")  # 「面」を削除
    assert keep == {0, 2}


def test_survive_insertion_warns_and_keeps():
    words = ["結局", "面倒"]
    keep, warns = render.surviving_words(words, "結局X面倒")
    assert keep == {0, 1}
    assert len(warns) == 1
    assert "追加" in warns[0]


def test_survive_replace_warns_and_keeps():
    words = ["結局", "面倒"]
    keep, warns = render.surviving_words(words, "結局面白")  # 倒→白
    assert keep == {0, 1}
    assert len(warns) == 1
    assert "書き換え" in warns[0]


def test_survive_english_spaced_words():
    words = [" hello", " um", " world"]
    keep, warns = render.surviving_words(words, " hello world")  # um削除
    assert keep == {0, 2}
    assert warns == []


# ---- words_to_intervals ----

def _w(start, end):
    return {"start": start, "end": end}


def test_intervals_merges_consecutive():
    words = [_w(0.0, 1.0), _w(1.0, 2.0), _w(2.0, 3.0)]
    ivs = render.words_to_intervals(words, {0, 1, 2}, margin=0.0, lo=0.0, hi=3.0)
    assert ivs == [(0.0, 3.0)]


def test_intervals_margin_applied():
    words = [_w(1.0, 2.0)]
    # 前後に十分な余地（lo=0, hi=10）→ フルにマージンが乗る
    ivs = render.words_to_intervals(words, {0}, margin=0.02, lo=0.0, hi=10.0)
    assert ivs == [(pytest.approx(0.98), pytest.approx(2.02))]


def test_intervals_no_intrusion_into_deleted_neighbors():
    # word1 を削除。word0 の end マージンは word1.start を越えない、
    # word2 の start マージンは word1.end を越えない
    words = [_w(0.0, 1.0), _w(1.0, 2.0), _w(2.0, 3.0)]
    ivs = render.words_to_intervals(words, {0, 2}, margin=0.5, lo=0.0, hi=3.0)
    assert ivs[0] == (0.0, pytest.approx(1.0))
    assert ivs[1] == (pytest.approx(2.0), 3.0)


def test_intervals_clamps_to_file_bounds():
    words = [_w(0.0, 1.0)]
    ivs = render.words_to_intervals(words, {0}, margin=0.5, lo=0.0, hi=1.2)
    assert ivs == [(0.0, pytest.approx(1.2))]


def test_intervals_empty_when_none_kept():
    words = [_w(0.0, 1.0)]
    assert render.words_to_intervals(words, set(), margin=0.02, hi=1.0) == []


# ---- crossfade_concat ----

def test_crossfade_length_reduced_by_fade():
    a = np.ones(100)
    b = np.ones(100)
    out = render.crossfade_concat([a, b], fade_samples=10)
    # 10サンプル重なる → 190
    assert len(out) == 190


def test_crossfade_no_click_on_sine():
    sr = 16000
    f = 220.0
    t = np.arange(sr) / sr
    sine = 0.5 * np.sin(2 * np.pi * f * t)
    # 2つの別位相区間を結合（境界で不連続が起きうる素材）
    a = sine[:4000]
    b = sine[8000:12000]
    fade = int(0.008 * sr)
    out = render.crossfade_concat([a, b], fade_samples=fade)
    # 境界（重なり終端）付近のサンプル間差分が過大でないこと
    diffs = np.abs(np.diff(out))
    assert diffs.max() < 0.05


def test_crossfade_empty():
    out = render.crossfade_concat([], fade_samples=10)
    assert len(out) == 0


def test_crossfade_stereo():
    a = np.ones((100, 2))
    b = np.ones((100, 2))
    out = render.crossfade_concat([a, b], fade_samples=10)
    assert out.shape == (190, 2)


# ---- end-to-end (合成wav + 手書き json/txt) ----

def _make_project(tmp_path, sr=16000):
    import soundfile as sf

    # 3語 x 1秒 = 3秒。各語で異なる周波数の sine
    seg = []
    freqs = [220.0, 440.0, 660.0]
    for fr in freqs:
        t = np.arange(sr) / sr
        seg.append(0.3 * np.sin(2 * np.pi * fr * t))
    audio = np.concatenate(seg)
    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "segments": [
            {
                "id": "0001",
                "text": "あいう",
                "words": [
                    {"word": "あ", "start": 0.0, "end": 1.0, "probability": 0.9},
                    {"word": "い", "start": 1.0, "end": 2.0, "probability": 0.9},
                    {"word": "う", "start": 2.0, "end": 3.0, "probability": 0.9},
                ],
            }
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_render_end_to_end_no_edit(tmp_path):
    import soundfile as sf

    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あいう\n", encoding="utf-8")

    out = render.render(str(txt))
    assert out.endswith("input.edited.wav")
    o_audio, o_sr = sf.read(out)
    assert o_sr == 16000
    # 無編集なら概ね元尺（3秒 ≒ 48000サンプル）に近い
    assert abs(len(o_audio) - 48000) < 2000


def test_render_end_to_end_delete_word(tmp_path):
    import soundfile as sf

    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あう\n", encoding="utf-8")  # 「い」削除

    out = render.render(str(txt))
    o_audio, o_sr = sf.read(out)
    # 1語(1秒)削除 → 約2秒に短縮
    assert abs(len(o_audio) - 32000) < 3000


def test_render_output_option(tmp_path):
    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あいう\n", encoding="utf-8")
    custom = tmp_path / "custom_out.wav"
    out = render.render(str(txt), output=str(custom))
    assert out == str(custom)
    assert custom.exists()


def test_render_subtype_preserved(tmp_path):
    import soundfile as sf

    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あいう\n", encoding="utf-8")
    out = render.render(str(txt))
    assert sf.info(out).subtype == "PCM_16"


def test_render_unknown_id_raises(tmp_path):
    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[9999] なにか\n", encoding="utf-8")
    with pytest.raises(ValueError):
        render.render(str(txt))


def test_render_missing_json_raises(tmp_path):
    txt = tmp_path / "orphan.sc.txt"
    txt.write_text("[0001] あ\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        render.render(str(txt))


def test_render_missing_txt_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        render.render(str(tmp_path / "nope.sc.txt"))


def test_render_missing_source_wav_raises(tmp_path):
    data = {
        "source_wav": str(tmp_path / "gone.wav"),
        "segments": [
            {"id": "0001", "text": "あ",
             "words": [{"word": "あ", "start": 0.0, "end": 1.0, "probability": 0.9}]}
        ],
    }
    (tmp_path / "x.sc.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    txt = tmp_path / "x.sc.txt"
    txt.write_text("[0001] あ\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        render.render(str(txt))


def test_render_warns_on_insertion(tmp_path, capsys):
    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あいうX\n", encoding="utf-8")  # 追加文字
    render.render(str(txt))
    err = capsys.readouterr().err
    assert "警告" in err


# ---- CLI end-to-end ----

def _make_multiseg_project(tmp_path, sr=16000):
    """3セグメント各1秒。json 上で時間軸連続。境界食い込み検証用。"""
    import soundfile as sf

    seg = []
    freqs = [220.0, 440.0, 660.0]
    for fr in freqs:
        t = np.arange(sr) / sr
        seg.append(0.3 * np.sin(2 * np.pi * fr * t))
    audio = np.concatenate(seg)
    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "segments": [
            {"id": "0001", "text": "あ",
             "words": [{"word": "あ", "start": 0.0, "end": 1.0, "probability": 0.9}]},
            {"id": "0002", "text": "い",
             "words": [{"word": "い", "start": 1.0, "end": 2.0, "probability": 0.9}]},
            {"id": "0003", "text": "う",
             "words": [{"word": "う", "start": 2.0, "end": 3.0, "probability": 0.9}]},
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_render_no_cross_segment_intrusion(tmp_path):
    """セグメント2を行削除。残す1・3のマージンが2の音声へ食い込まないこと。"""
    import soundfile as sf

    _make_multiseg_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ\n[0003] う\n", encoding="utf-8")  # 0002 を行削除

    out = render.render(str(txt))
    o_audio, o_sr = sf.read(out)
    # seg1: [0,1] のみ（後マージンは 0002 の start=1.0 でクランプ）
    # seg3: [2,3] のみ（前マージンは 0002 の end=2.0 でクランプ）
    # クロスフェード無しなら約2秒(32000)。フェード重なりで僅かに短い。
    # マージンが 0002 に食い込むと長くなる → 上限で検証。
    fade = int(0.008 * 16000)
    assert len(o_audio) <= 32000
    assert len(o_audio) >= 32000 - fade - 5


def test_render_edge_segment_margin_uses_file_bounds(tmp_path):
    """先頭/末尾セグメントは前後が無いのでマージンがファイル範囲内で乗る。"""
    import soundfile as sf

    _make_multiseg_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    # 0002 のみ残す → lo=0002 の前 0001.end=1.0, hi=0003.start=2.0
    txt.write_text("[0002] い\n", encoding="utf-8")
    out = render.render(str(txt))
    o_audio, _ = sf.read(out)
    # [1.0,2.0] のみ、前後マージンは隣接単語境界でクランプ → 約1秒(16000)
    assert abs(len(o_audio) - 16000) < 200


def test_cli_render(tmp_path):
    from stefnceorf import cli

    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あいう\n", encoding="utf-8")
    rc = cli.main(["render", str(txt)])
    assert rc == 0
    assert (tmp_path / "input.edited.wav").exists()


def test_cli_render_error(tmp_path):
    from stefnceorf import cli

    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[9999] x\n", encoding="utf-8")
    rc = cli.main(["render", str(txt)])
    assert rc == 1
