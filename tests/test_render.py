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


def test_parse_strips_block_separator():
    # ／（ブロック区切り）は除去される
    txt = "[0001] 結局／面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒")]


# ---- snap_to_blocks ----

def test_snap_partial_delete_removes_block():
    # block0={0,1}, block1={2}。word1 削除 → block0 全削除、extra に word0
    keep, extra = render.snap_to_blocks({0, 2}, [0, 0, 1])
    assert keep == {2}
    assert extra == {0}


def test_snap_all_survive_unchanged():
    keep, extra = render.snap_to_blocks({0, 1, 2}, [0, 0, 1])
    assert keep == {0, 1, 2}
    assert extra == set()


def test_snap_identity_blocks_passthrough():
    # 各単語が独立ブロック → 入力そのまま
    keep, extra = render.snap_to_blocks({0, 2}, [0, 1, 2])
    assert keep == {0, 2}
    assert extra == set()


def test_snap_empty_keep():
    keep, extra = render.snap_to_blocks(set(), [0, 0, 1])
    assert keep == set()
    assert extra == set()


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


def test_survive_punct_diff_ambiguity_no_dragging():
    """読点付きトークンの diff 曖昧性で隣の単語を巻き込まない（回帰テスト）。

    「…を、」＋「あの、」から「あの、」を削除すると、difflib は削除を
    「、あの」に整列することがある。句読点は生存判定から除外するため
    「生成AIを、」は巻き込まれない。
    """
    words = ["生成AIが", "生成AIを、", "あの、", "最先端の"]
    keep, warns = render.surviving_words(words, "生成AIが生成AIを、最先端の")
    assert keep == {0, 1, 3}


def test_survive_punct_only_word_always_kept():
    # 句読点のみの単語は判定対象文字がなく常に生存扱い（空白のみと同様）
    words = ["結局", "、", "面倒"]
    keep, _ = render.surviving_words(words, "結局、面倒")
    assert keep == {0, 1, 2}


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
    # 前後に十分な余地（lo=0, hi=10）→ start は margin、end は tail_margin まで拡張
    ivs = render.words_to_intervals(words, {0}, margin=0.02, tail_margin=0.02,
                                     lo=0.0, hi=10.0)
    assert ivs == [(pytest.approx(0.98), pytest.approx(2.02))]


def test_intervals_tail_margin_extends_end():
    words = [_w(1.0, 2.0)]
    # tail_margin=0.2 で後方に余地あり → end は 2.0+0.2=2.2 まで拡張
    ivs = render.words_to_intervals(words, {0}, margin=0.02, tail_margin=0.2,
                                     lo=0.0, hi=10.0)
    assert ivs == [(pytest.approx(0.98), pytest.approx(2.2))]


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


# ---- plan_output_intervals（ギャップ保持/切り詰め 純関数） ----

def test_plan_single_interval_unchanged():
    out, flags = render.plan_output_intervals([(0.0, 1.0)], [])
    assert out == [(0.0, 1.0)]
    assert flags == []


def test_plan_short_gap_kept_merges():
    # ギャップ 0.5s（<1.5）で純無音 → 連続区間にマージ
    ivs = [(0.0, 1.0), (1.5, 2.5)]
    out, flags = render.plan_output_intervals(ivs, [], gap_threshold=1.5, gap_max=0.7)
    assert out == [(0.0, 2.5)]
    assert flags == []


def test_plan_contiguous_merges():
    # g==0（隙間なし）でも純無音扱い → 連続マージ
    ivs = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    out, flags = render.plan_output_intervals(ivs, [], gap_threshold=1.5, gap_max=0.7)
    assert out == [(0.0, 3.0)]
    assert flags == []


def test_plan_long_gap_trimmed():
    # ギャップ 3.0s（>1.5）で純無音 → 0.7s に切り詰め、単純結合
    ivs = [(0.0, 1.0), (4.0, 5.0)]
    out, flags = render.plan_output_intervals(ivs, [], gap_threshold=1.5, gap_max=0.7)
    assert out == [
        (0.0, pytest.approx(1.35)),
        (pytest.approx(3.65), 5.0),
    ]
    assert flags == [False]


def test_plan_deleted_word_in_gap_not_merged():
    # ギャップ [1.0,4.0] に削除単語 [1.5,3.5] が挟まる → クロスフェード結合
    ivs = [(0.0, 1.0), (4.0, 5.0)]
    out, flags = render.plan_output_intervals(
        ivs, [(1.5, 3.5)], gap_threshold=1.5, gap_max=0.7
    )
    assert out == [(0.0, 1.0), (4.0, 5.0)]
    assert flags == [True]


def test_plan_reverse_join_not_merged():
    # 並べ替えでソース逆行（g<0）→ クロスフェード結合
    ivs = [(4.0, 5.0), (0.0, 1.0)]
    out, flags = render.plan_output_intervals(ivs, [], gap_threshold=1.5, gap_max=0.7)
    assert out == [(4.0, 5.0), (0.0, 1.0)]
    assert flags == [True]


def test_plan_kept_neighbor_words_do_not_block_merge():
    # 隣接単語自身（区間端の外側）は間の単語とみなされない → マージされる
    ivs = [(0.0, 1.0), (1.5, 2.5)]
    spans = [(0.0, 0.98), (1.52, 2.5)]
    out, flags = render.plan_output_intervals(ivs, spans, gap_threshold=1.5, gap_max=0.7)
    assert out == [(0.0, 2.5)]
    assert flags == []


def test_crossfade_concat_no_crossfade_flag():
    """crossfade_flags=False の境界では単純結合（長さ減少なし）"""
    a = np.ones(100)
    b = np.ones(100)
    out = render.crossfade_concat([a, b], fade_samples=10, crossfade_flags=[False])
    assert len(out) == 200


def test_crossfade_concat_mixed_flags():
    """境界ごとにクロスフェード/単純結合を切り替え"""
    a = np.ones(100)
    b = np.ones(100)
    c = np.ones(100)
    out = render.crossfade_concat([a, b, c], fade_samples=10, crossfade_flags=[True, False])
    # a-b: crossfade (190), b-c: simple concat (+100)
    assert len(out) == 290


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


def _make_block_project(tmp_path, sr=16000):
    """3語 x 1秒。blocks=[0,0,1]（「あ」「い」が同ブロック、「う」が別ブロック）。"""
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
        "pause_threshold": 0.15,
        "segments": [
            {
                "id": "0001",
                "text": "あいう",
                "words": [
                    {"word": "あ", "start": 0.0, "end": 1.0, "probability": 0.9, "block": 0},
                    {"word": "い", "start": 1.0, "end": 2.0, "probability": 0.9, "block": 0},
                    {"word": "う", "start": 2.0, "end": 3.0, "probability": 0.9, "block": 1},
                ],
            }
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_render_block_snap_removes_whole_block(tmp_path):
    """ブロック内1語削除 → 同ブロックの2語とも音声から消える（尺で検証）。"""
    import soundfile as sf

    _make_block_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    # block0=「あい」の「い」を削除 → block0 全体（あい=2秒）が消え「う」1秒のみ残る
    txt.write_text("[0001] あ／う\n", encoding="utf-8")

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    assert abs(len(o_audio) - 16000) < 3000


def test_render_block_snap_warns(tmp_path, capsys):
    """スナップで巻き込まれた語が警告に列挙される。"""
    _make_block_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ／う\n", encoding="utf-8")
    render.render(str(txt))
    err = capsys.readouterr().err
    assert "ポーズ区切りに合わせて追加削除" in err
    assert "あ" in err


def test_render_block_other_block_kept(tmp_path):
    """別ブロックは残る（block1「う」を消しても block0「あい」は残る）。"""
    import soundfile as sf

    _make_block_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あい\n", encoding="utf-8")  # 「う」削除

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    # block0（あい=2秒）残存
    assert abs(len(o_audio) - 32000) < 3000


def test_render_block_separator_kept_equals_no_edit(tmp_path):
    """／を残したまま render は無編集と同一結果になる。"""
    import soundfile as sf

    _make_block_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あい／う\n", encoding="utf-8")
    out = render.render(str(txt), output=str(tmp_path / "kept.wav"))
    a, _ = sf.read(out)

    txt.write_text("[0001] あいう\n", encoding="utf-8")
    out2 = render.render(str(txt), output=str(tmp_path / "plain.wav"))
    b, _ = sf.read(out2)
    assert len(a) == len(b)


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


def _make_gap_project(tmp_path, sr=16000):
    """3セグメント。seg1[0-1], 長無音(3s), seg2[4-5], 短ポーズ(0.5s), seg3[5.5-6.5]。

    無音区間には単語が無い純無音。全長 6.5s。
    """
    import soundfile as sf

    audio = np.zeros(int(6.5 * sr))
    specs = [(0.0, 1.0, 220.0), (4.0, 5.0, 440.0), (5.5, 6.5, 660.0)]
    for s, e, fr in specs:
        t = np.arange(int((e - s) * sr)) / sr
        audio[int(s * sr): int(s * sr) + len(t)] = 0.3 * np.sin(2 * np.pi * fr * t)
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
             "words": [{"word": "い", "start": 4.0, "end": 5.0, "probability": 0.9}]},
            {"id": "0003", "text": "う",
             "words": [{"word": "う", "start": 5.5, "end": 6.5, "probability": 0.9}]},
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_render_gap_trim_and_keep(tmp_path):
    """既定: 長無音は0.7sに切り詰め、短ポーズは保持される。"""
    import soundfile as sf

    _make_gap_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ\n[0002] い\n[0003] う\n", encoding="utf-8")

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    # 期待: seg1(≈1.22) + トリム無音(≈0.7) + seg2〜seg3を短ポーズ込みで連続(≈2.87)
    # ≈ 4.42s。tail_margin=0.2 で後方マージンが拡張されている。
    dur = len(o_audio) / sr
    assert dur == pytest.approx(4.42, abs=0.15)


def test_render_gap_threshold_keeps_all(tmp_path):
    """--gap-threshold を大きくすると長無音も保持され元尺に近づく。"""
    import soundfile as sf

    _make_gap_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ\n[0002] い\n[0003] う\n", encoding="utf-8")

    out = render.render(str(txt), gap_threshold=100.0)
    o_audio, sr = sf.read(out)
    assert len(o_audio) / sr == pytest.approx(6.5, abs=0.05)


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
