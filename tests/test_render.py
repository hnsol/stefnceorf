import json

import numpy as np
import pytest

from stefnceorf import render


# ---- parse_edited_txt ----

def test_parse_plain():
    txt = "[0001] 結局面倒\n[0002] 作業\n"
    assert render.parse_edited_txt(txt) == [
        ("0001", "結局面倒", []), ("0002", "作業", [])
    ]


def test_parse_with_timestamp():
    """新形式 [ID 時刻] はIDのみ抽出し、テキストは正常にパースされる。"""
    txt = "[0001 0:12] 結局面倒\n[0002 12:34] 作業\n"
    assert render.parse_edited_txt(txt) == [
        ("0001", "結局面倒", []), ("0002", "作業", [])
    ]


def test_parse_timestamp_hhmmss():
    """H:MM:SS 形式の時刻付き行もIDのみ抽出できる。"""
    txt = "[0001 1:02:03] テスト\n"
    assert render.parse_edited_txt(txt) == [("0001", "テスト", [])]


def test_parse_legacy_and_new_mixed():
    """旧形式と新形式が混在してもそれぞれ正しくパースされる。"""
    txt = "[0001] 旧形式\n[0002 0:05] 新形式\n"
    assert render.parse_edited_txt(txt) == [
        ("0001", "旧形式", []), ("0002", "新形式", [])
    ]


def test_parse_ignores_blank_lines():
    txt = "[0001] あ\n\n   \n[0002] い\n"
    assert render.parse_edited_txt(txt) == [("0001", "あ", []), ("0002", "い", [])]


def test_parse_strips_low_conf_mark():
    txt = "[0001] 結局◆面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒", [])]


def test_parse_filler_bracket_deleted():
    # 〔...〕 は中身ごと削除。中身は出現順にリストで返る
    txt = "[0001] 結局〔まあ〕面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒", ["まあ"])]


def test_parse_filler_bracket_removed_keeps_text():
    # 括弧を外した箇所は通常文字として残る
    txt = "[0001] 結局まあ面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局まあ面倒", [])]


def test_parse_multiple_brackets_in_order():
    # 複数の〔〕は出現順に中身が集まる
    txt = "[0001] 〔あの、〕結局〔まあ〕面倒\n"
    assert render.parse_edited_txt(txt) == [
        ("0001", "結局面倒", ["あの、", "まあ"])
    ]


def test_parse_bracket_content_strips_markers():
    # 〔〕中身に混在した ◆・／ は中身から除去される
    txt = "[0001] 結局〔◆まあ〕面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒", ["まあ"])]


def test_clean_edited_text_returns_contents():
    cleaned, brackets = render._clean_edited_text("結局〔まあ〕面倒")
    assert cleaned == "結局面倒"
    assert brackets == ["まあ"]


def test_parse_broken_id_line_raises():
    with pytest.raises(ValueError):
        render.parse_edited_txt("結局面倒\n")


def test_parse_duplicate_id_raises():
    with pytest.raises(ValueError):
        render.parse_edited_txt("[0001] あ\n[0001] い\n")


def test_parse_strips_block_separator():
    # ／（ブロック区切り）は除去される
    txt = "[0001] 結局／面倒\n"
    assert render.parse_edited_txt(txt) == [("0001", "結局面倒", [])]


# ---- match_filler_deletions ----

from stefnceorf import fillers as fillers_mod

_JA = fillers_mod.load_fillers("ja")


def _sw(word, suggest=False):
    d = {"word": word, "start": 0.0, "end": 1.0, "probability": 0.9}
    if suggest:
        d["suggest"] = True
    return d


def test_match_filler_ordered():
    # 提案単語2つ、〔〕2つが順序どおりマッチ
    words = [_sw("結局"), _sw("まあ", True), _sw("面倒"), _sw("えー", True)]
    filler_del, warns = render.match_filler_deletions(words, ["まあ", "えー"], _JA)
    assert filler_del == {1, 3}
    assert warns == []


def test_match_filler_same_content_partial_kept():
    # 同一内容〔あの、〕が2つ提案されるが片方だけ残された → 順序どおり先頭にマッチ
    words = [_sw("あの、", True), _sw("結局"), _sw("あの、", True), _sw("面倒")]
    filler_del, warns = render.match_filler_deletions(words, ["あの、"], _JA)
    assert filler_del == {0}
    assert warns == []


def test_match_filler_unmatched_warns():
    # 対応する提案単語が無い → 警告し diff 委譲（削除確定には含めない）
    words = [_sw("結局"), _sw("まあ", True)]
    filler_del, warns = render.match_filler_deletions(words, ["えー"], _JA)
    assert filler_del == set()
    assert len(warns) == 1
    assert "えー" in warns[0]


def test_match_filler_legacy_json_fallback():
    # suggest キーが1つも無い旧 json → is_filler フォールバックでマッチ
    words = [_sw("結局"), _sw("まあ"), _sw("面倒")]
    filler_del, warns = render.match_filler_deletions(words, ["まあ"], _JA)
    assert filler_del == {1}
    assert warns == []


def test_match_filler_empty_brackets():
    words = [_sw("結局"), _sw("まあ", True)]
    assert render.match_filler_deletions(words, [], _JA) == (set(), [])


def test_match_filler_relaxed_normalize():
    # 完全一致しないが normalize_token 同士は一致（読点差）
    words = [_sw("まあ、", True)]
    filler_del, warns = render.match_filler_deletions(words, ["まあ"], _JA)
    assert filler_del == {0}
    assert warns == []


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


# ---- snap_with_filler_exemption ----

def test_snap_exemption_filler_only_keeps_block_survivor():
    # block0={0,1}。word1 はフィラー削除確定 → word0 は生存（ブロック巻き込み無し）
    keep, extra = render.snap_with_filler_exemption({0}, [0, 0, 1], {1})
    assert keep == {0}
    assert extra == set()


def test_snap_exemption_filler_plus_normal_delete_wipes_block():
    # block0={0,1,2}。word2 は通常削除（keep にも filler_del にも無い）→
    # ブロック全体が従来スナップで全滅する。extra は snap_to_blocks の値を
    # そのまま返すので keep|filler_del の全要素（{0,1}）を含む。
    keep, extra = render.snap_with_filler_exemption({0}, [0, 0, 0], {1})
    assert keep == set()
    assert extra == {0, 1}


def test_snap_exemption_empty_filler_equals_snap_to_blocks():
    keep_a, extra_a = render.snap_with_filler_exemption({0, 2}, [0, 0, 1], set())
    keep_b, extra_b = render.snap_to_blocks({0, 2}, [0, 0, 1])
    assert (keep_a, extra_a) == (keep_b, extra_b)


# ---- boundary_rms ----

def test_boundary_rms_silence_very_low():
    audio = np.zeros(16000)
    assert render.boundary_rms(audio, 16000, 0.5) == float("-inf")


def test_boundary_rms_sine_about_minus_9db():
    sr = 16000
    t = np.arange(sr) / sr
    sine = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    db = render.boundary_rms(sine, sr, 0.5, half_window_s=0.05)
    # 振幅0.5正弦波の RMS = 0.5/sqrt(2) ≈ 0.354 → 20log10 ≈ -9.03 dBFS
    assert db == pytest.approx(-9.03, abs=0.3)


def test_boundary_rms_clamps_to_file_edge():
    sr = 16000
    t = np.arange(sr) / sr
    sine = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    # t=0 でも窓は [0, +half] にクランプされ計算可能
    db = render.boundary_rms(sine, sr, 0.0)
    assert db > -20.0


def test_boundary_rms_stereo_channel_mean():
    sr = 16000
    t = np.arange(sr) / sr
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    d_mono = render.boundary_rms(mono, sr, 0.5, half_window_s=0.05)
    d_stereo = render.boundary_rms(stereo, sr, 0.5, half_window_s=0.05)
    assert d_stereo == pytest.approx(d_mono, abs=0.01)


# ---- filler_cut_is_safe ----

def _sine(sr, freq, dur, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_filler_cut_safe_both_boundaries_silent():
    sr = 16000
    audio = np.zeros(3 * sr)  # 全無音
    word = {"start": 1.0, "end": 2.0}
    assert render.filler_cut_is_safe(audio, sr, word, silences=None) is True


def test_filler_cut_unsafe_when_boundary_continuous_sine():
    sr = 16000
    audio = _sine(sr, 220.0, 3.0)  # 連続高RMS
    word = {"start": 1.0, "end": 2.0}
    assert render.filler_cut_is_safe(audio, sr, word, silences=None) is False


def test_filler_cut_safe_via_silence_edge_tolerance():
    sr = 16000
    audio = _sine(sr, 220.0, 3.0)  # RMS は高いが silences 端で安全
    word = {"start": 1.0, "end": 2.0}
    # 端±10ms 一致（1.0 と 2.0 が各 silence の端に接する）
    silences = [[0.5, 0.995], [2.005, 2.5]]
    assert render.filler_cut_is_safe(audio, sr, word, silences=silences) is True


def test_filler_cut_silences_none_uses_rms_only():
    sr = 16000
    audio = np.zeros(3 * sr)
    word = {"start": 1.0, "end": 2.0}
    # silences=None でも RMS が低ければ安全
    assert render.filler_cut_is_safe(audio, sr, word, silences=None) is True


def test_filler_cut_start_none_is_unsafe():
    sr = 16000
    audio = np.zeros(3 * sr)
    word = {"start": None, "end": 2.0}
    assert render.filler_cut_is_safe(audio, sr, word, silences=None) is False


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


# ---- plan_output_intervals（フィラーポーズ分岐） ----

def test_plan_filler_gap_keeps_pause_total():
    # gap [1.0,3.0] にフィラー span [1.0,3.0] のみ（両端に十分な余地）
    ivs = [(0.0, 1.0), (3.0, 4.0)]
    out, flags = render.plan_output_intervals(
        ivs, [], filler_spans=[(1.5, 2.5)], filler_pause_keep=0.25
    )
    # pre 側 f_start=1.5, post 側 f_end=2.5。pre_keep=post_keep=0.125
    assert out == [
        (0.0, pytest.approx(1.125)),
        (pytest.approx(2.875), 4.0),
    ]
    assert flags == [False]
    # 残る間の合計 = (1.125-1.0) + (3.0-2.875) = 0.25
    total = (out[0][1] - 1.0) + (3.0 - out[1][0])
    assert total == pytest.approx(0.25)


def test_plan_filler_pre_short_redistributes_to_post():
    # フィラー span が pe に密着（pre_avail 小）→ 余りが post へ再配分
    ivs = [(0.0, 1.0), (3.0, 4.0)]
    out, flags = render.plan_output_intervals(
        ivs, [], filler_spans=[(1.05, 2.0)], filler_pause_keep=0.25
    )
    # pre_avail=0.05, post_avail=1.0。pre_keep=0.05, post_keep=0.125+0.075=0.2
    assert out[0][1] == pytest.approx(1.05)
    assert out[1][0] == pytest.approx(3.0 - 0.2)
    total = (out[0][1] - 1.0) + (3.0 - out[1][0])
    assert total == pytest.approx(0.25)
    assert flags == [False]


def test_plan_filler_both_short_caps_at_available():
    # 両側の avail 合計 < keep → 合計は avail 合計まで
    ivs = [(0.0, 1.0), (1.15, 2.15)]
    out, flags = render.plan_output_intervals(
        ivs, [], filler_spans=[(1.05, 1.10)], filler_pause_keep=0.25
    )
    # pre_avail=0.05, post_avail=0.05 → 合計 0.10
    total = (out[0][1] - 1.0) + (1.15 - out[1][0])
    assert total == pytest.approx(0.10)
    assert flags == [False]


def test_plan_filler_with_normal_delete_prefers_crossfade():
    # フィラー span と通常削除語が同 gap → 分岐2（単語挟まり）優先でクロスフェード
    ivs = [(0.0, 1.0), (4.0, 5.0)]
    out, flags = render.plan_output_intervals(
        ivs, [(1.5, 3.5)], filler_spans=[(1.2, 1.4)], filler_pause_keep=0.25
    )
    assert out == [(0.0, 1.0), (4.0, 5.0)]
    assert flags == [True]


def test_plan_filler_none_matches_existing():
    # filler_spans=None は既存挙動（純無音の長ギャップ切り詰め）
    ivs = [(0.0, 1.0), (4.0, 5.0)]
    out, flags = render.plan_output_intervals(
        ivs, [], gap_threshold=1.5, gap_max=0.7, filler_spans=None
    )
    assert out == [
        (0.0, pytest.approx(1.35)),
        (pytest.approx(3.65), 5.0),
    ]
    assert flags == [False]


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


def _make_ambiguous_filler_project(tmp_path, sr=16000):
    """フィラー「まあ」の文字列が隣接語「まあね」の一部にも含まれる構成。

    w0=結局[0,1](220Hz), w1=まあ[1,2](440Hz, suggest, 単独ブロック),
    w2=まあね[2,2.5](660Hz)。従来の文字diffだと「まあ」削除が w1 と w2 の
    どちらの「まあ」に整列するか曖昧になる。w1/w2 の長さを変え（1.0s/0.5s）、
    削除された語を出力尺で判別できるようにする。
    """
    import soundfile as sf

    specs = [(0.0, 1.0, 220.0), (1.0, 2.0, 440.0), (2.0, 2.5, 660.0)]
    audio = np.zeros(int(2.5 * sr))
    for s, e, fr in specs:
        t = np.arange(int((e - s) * sr)) / sr
        audio[int(s * sr): int(s * sr) + len(t)] = 0.3 * np.sin(2 * np.pi * fr * t)
    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "pause_threshold": 0.15,
        # まあ[1,2] の両境界に無音を用意し音響安全判定を通す（連続音でないと
        # みなさせ、精密カットを許可する。安全判定は§4c）。
        "silences": [[0.98, 1.02], [1.98, 2.02]],
        "segments": [
            {
                "id": "0001",
                "text": "結局まあまあね",
                "words": [
                    {"word": "結局", "start": 0.0, "end": 1.0,
                     "probability": 0.9, "block": 0},
                    {"word": "まあ", "start": 1.0, "end": 2.0,
                     "probability": 0.9, "block": 1, "suggest": True},
                    {"word": "まあね", "start": 2.0, "end": 2.5,
                     "probability": 0.9, "block": 2},
                ],
            }
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_render_filler_bracket_deletes_exact_word(tmp_path):
    """〔まあ〕削除が、文字列が重なる隣接語「まあね」を巻き込まず w1 だけ消す。"""
    import soundfile as sf

    _make_ambiguous_filler_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    # 〔まあ〕を残す＝提案受け入れ（w1 を削除）。w2「まあね」は残す。
    txt.write_text("[0001] 結局〔まあ〕まあね\n", encoding="utf-8")

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    # w1(1.0s) 削除 → 結局(1.0s)+まあね(0.5s) = 1.5s。
    # 誤って w2(0.5s) を消すと 2.0s になるため 1.5s 近傍で判別できる。
    dur = len(o_audio) / sr
    assert dur == pytest.approx(1.5, abs=0.1)


def test_render_filler_bracket_unmatched_falls_back(tmp_path, capsys):
    """提案単語に無い〔〕内容は警告し、削除自体は diff に委ねられる。"""
    import soundfile as sf

    _make_ambiguous_filler_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    # 〔えー〕は提案単語に無い → 警告。テキスト本体は結局まあまあね（無削除相当）
    txt.write_text("[0001] 結局まあまあね〔えー〕\n", encoding="utf-8")

    render.render(str(txt))
    err = capsys.readouterr().err
    assert "えー" in err
    assert "見つかりません" in err


def test_render_warns_on_insertion(tmp_path, capsys):
    _make_project(tmp_path)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あいうX\n", encoding="utf-8")  # 追加文字
    render.render(str(txt))
    err = capsys.readouterr().err
    assert "警告" in err


# ---- end-to-end: フィラー精密カット（安全判定・片側ポーズ残し・手動削除） ----

def _write_words_project(tmp_path, specs, silences=None, sr=16000):
    """word 仕様列から wav + json を書く。

    specs: [(word, start, end, freq, block, suggest_bool), ...]。
    audio は全体を無音(zeros)で作り、各 word の [start,end] に sine を書く
    （word 間に無音の gap を作れる）。silences は json に付与する。
    """
    import soundfile as sf

    total = max(e for _, _, e, _, _, _ in specs)
    audio = np.zeros(int(round(total * sr)))
    words = []
    text = ""
    for word, s, e, fr, block, suggest in specs:
        t = np.arange(int(round((e - s) * sr))) / sr
        a = int(round(s * sr))
        audio[a: a + len(t)] = 0.3 * np.sin(2 * np.pi * fr * t)
        wd = {"word": word, "start": s, "end": e, "probability": 0.9,
              "block": block}
        if suggest:
            wd["suggest"] = True
        words.append(wd)
        text += word
    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "pause_threshold": 0.15,
        "segments": [{"id": "0001", "text": text, "words": words}],
    }
    if silences is not None:
        data["silences"] = silences
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_render_filler_pause_precise_cut(tmp_path, capsys):
    """前後に実無音を持つフィラーの〔〕削除 → フィラーだけ消え、接合部に
    約0.25秒の間が残り、隣接語は残存する。"""
    import soundfile as sf

    # 結局[0,1]220 / 無音[1,1.3] / まあ[1.3,2.0]440(filler) / 無音[2,2.3] / 面倒[2.3,3.3]660
    _write_words_project(
        tmp_path,
        [("結局", 0.0, 1.0, 220.0, 0, False),
         ("まあ", 1.3, 2.0, 440.0, 1, True),
         ("面倒", 2.3, 3.3, 660.0, 2, False)],
        silences=[[1.0, 1.3], [2.0, 2.3]],
    )
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] 結局〔まあ〕面倒\n", encoding="utf-8")

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    dur = len(o_audio) / sr
    # 出力 = [0,1.3] + [2.13,3.3] = 2.47s（pre_keep 0.1 + post_keep 0.15 = 0.25 の
    # ポーズ残し込み）。フィラーを通常削除扱いで詰めると約2.22s になるため区別できる。
    assert dur == pytest.approx(2.47, abs=0.08)
    err = capsys.readouterr().err
    assert "残しました" not in err


def test_render_filler_continuous_audio_kept(tmp_path, capsys):
    """フィラーが隣接語と連続音（無音なし・高RMS）→ 尺が減らず警告が出る。"""
    import soundfile as sf

    _write_words_project(
        tmp_path,
        [("結局", 0.0, 1.0, 220.0, 0, False),
         ("まあ", 1.0, 2.0, 440.0, 1, True),
         ("面倒", 2.0, 3.0, 660.0, 2, False)],
        silences=None,  # 無音区間情報なし・境界は連続高RMS
    )
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] 結局〔まあ〕面倒\n", encoding="utf-8")

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    dur = len(o_audio) / sr
    # フィラーは安全でないため残る → ほぼ元尺（3.0s）
    assert dur == pytest.approx(3.0, abs=0.1)
    err = capsys.readouterr().err
    assert "残しました" in err
    assert "まあ" in err


def test_render_manual_filler_delete_no_block_wipe(tmp_path, capsys):
    """手動削除フィラー（〔〕でなく文字を直接消す・同ブロックに他語あり）→
    ブロック巻き込みなしで該当フィラーのみ消える。"""
    import soundfile as sf

    # 結局 と まあ が同ブロック(0)。まあ の両境界には無音があり安全にカット可能。
    _write_words_project(
        tmp_path,
        [("結局", 0.0, 1.0, 220.0, 0, False),
         ("まあ", 1.3, 2.0, 440.0, 0, False),
         ("面倒", 2.3, 3.3, 660.0, 1, False)],
        silences=[[1.0, 1.3], [2.0, 2.3]],
    )
    txt = tmp_path / "input.sc.txt"
    # 〔〕でなく「まあ」の文字を直接削除（＝結局面倒）
    txt.write_text("[0001] 結局面倒\n", encoding="utf-8")

    out = render.render(str(txt))
    o_audio, sr = sf.read(out)
    dur = len(o_audio) / sr
    # 4a により まあ のみ精密カット、同ブロックの 結局 は残存 → 約2.47s。
    # ブロック巻き込みで 結局 まで消えると約1.2s になるため区別できる。
    assert dur == pytest.approx(2.47, abs=0.08)
    err = capsys.readouterr().err
    assert "ポーズ区切りに合わせて追加削除" not in err
    assert "残しました" not in err


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
