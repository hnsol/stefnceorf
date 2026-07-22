import json
import sys
import types

import pytest

from stefnceorf import transcribe
from stefnceorf import fillers as fillers_mod


# ---- plan_chunks（純関数） ----

def test_plan_chunks_chooses_longest_silence_near_target():
    assert transcribe.plan_chunks(
        550.0, [(260.0, 262.0), (320.0, 330.0)]
    ) == [(0.0, 325.0), (325.0, 550.0)]


def test_plan_chunks_short_audio_is_single_chunk():
    assert transcribe.plan_chunks(350.0, []) == [(0.0, 350.0)]


def test_plan_chunks_without_silence_uses_upper_boundary():
    assert transcribe.plan_chunks(700.0, []) == [
        (0.0, 360.0), (360.0, 700.0)
    ]


def test_plan_chunks_has_no_gap_or_overlap():
    chunks = transcribe.plan_chunks(
        1200.0, [(295.0, 305.0), (600.0, 610.0)]
    )
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 1200.0
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


# ---- _preserve_unrecognized_gaps（主認識の疎な抜けを音声保持） ----

def _speech_segment(start: float, end: float, text: str = "発話") -> dict:
    return {
        "text": text,
        "words": [
            {
                "word": text,
                "start": start,
                "end": end,
                "probability": 0.9,
            }
        ],
    }


def test_preserve_unrecognized_gap_without_known_silence():
    segments = [_speech_segment(0.0, 1.0), _speech_segment(4.0, 5.0)]

    out = transcribe._preserve_unrecognized_gaps(
        segments, cuts=[], silences=[], source_duration=5.0
    )

    assert [seg.get("kind", "speech") for seg in out] == [
        "speech", "unrecognized", "speech"
    ]
    assert (out[1]["source_start"], out[1]["source_end"]) == (1.0, 4.0)
    assert segments == [_speech_segment(0.0, 1.0), _speech_segment(4.0, 5.0)]


def test_preserve_gap_fully_covered_by_known_silence_does_not_add_marker():
    segments = [_speech_segment(0.0, 1.0), _speech_segment(4.0, 5.0)]

    out = transcribe._preserve_unrecognized_gaps(
        segments, cuts=[], silences=[(1.0, 4.0)], source_duration=5.0
    )

    assert out == segments


def test_preserve_gap_partly_covered_by_silence_keeps_whole_gap():
    out = transcribe._preserve_unrecognized_gaps(
        [_speech_segment(0.0, 1.0), _speech_segment(4.0, 5.0)],
        cuts=[],
        silences=[(1.0, 3.9)],
        source_duration=5.0,
    )

    assert (out[1]["source_start"], out[1]["source_end"]) == (1.0, 4.0)


def test_preserve_non_silence_file_edges_regardless_of_length():
    out = transcribe._preserve_unrecognized_gaps(
        [_speech_segment(0.2, 4.8)],
        cuts=[],
        silences=[],
        source_duration=5.0,
    )

    assert [seg.get("kind", "speech") for seg in out] == [
        "unrecognized", "speech", "unrecognized"
    ]
    assert (out[0]["source_start"], out[0]["source_end"]) == (0.0, 0.2)
    assert (out[2]["source_start"], out[2]["source_end"]) == (4.8, 5.0)


def test_preserve_gap_across_hard_chunk_boundary():
    out = transcribe._preserve_unrecognized_gaps(
        [_speech_segment(358.0, 359.0), _speech_segment(362.0, 363.0)],
        cuts=[],
        silences=[],
        source_duration=363.0,
    )

    gaps = [seg for seg in out if seg.get("kind") == "unrecognized"]
    assert [(seg["source_start"], seg["source_end"]) for seg in gaps] == [
        (0.0, 358.0),
        (359.0, 362.0),
    ]


def test_preserve_whole_source_when_recognition_is_empty():
    out = transcribe._preserve_unrecognized_gaps(
        [], cuts=[], silences=[], source_duration=10.0
    )

    assert out == [
        {
            "kind": "unrecognized",
            "source_start": 0.0,
            "source_end": 10.0,
            "text": "",
            "words": [],
        }
    ]


def test_preserve_existing_unrecognized_coverage_without_overlap():
    existing = {
        "kind": "unrecognized",
        "source_start": 1.0,
        "source_end": 4.0,
        "text": "既存",
        "words": [],
    }
    segments = [_speech_segment(0.0, 1.0), existing, _speech_segment(4.0, 5.0)]

    out = transcribe._preserve_unrecognized_gaps(
        segments, cuts=[], silences=[], source_duration=5.0
    )

    assert out == segments


def test_preserve_coverage_uses_final_short_silence_clamped_word_bounds():
    segment = _speech_segment(0.0, 1.5)

    out = transcribe._preserve_unrecognized_gaps(
        [segment],
        cuts=[],
        silences=[(1.2, 1.8)],
        known_silences=[],
        source_duration=2.0,
    )

    assert [seg.get("kind", "speech") for seg in out] == [
        "speech", "unrecognized"
    ]
    assert (out[1]["source_start"], out[1]["source_end"]) == (1.2, 2.0)


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


# ---- assign_blocks（純関数） ----

def _bw(start, end):
    return {"word": "x", "start": start, "end": end, "probability": 0.9}


def test_assign_blocks_empty():
    assert transcribe.assign_blocks([], [], 0.15) == []


def test_assign_blocks_threshold_zero_all_boundaries():
    words = [_bw(0.0, 1.0), _bw(1.0, 2.0), _bw(2.0, 3.0)]
    assert transcribe.assign_blocks(words, [], 0) == [0, 1, 2]


def test_assign_blocks_no_silence_contiguous_single_block():
    # ギャップ0・無音なし → 全て同一ブロック
    words = [_bw(0.0, 1.0), _bw(1.0, 2.0), _bw(2.0, 3.0)]
    assert transcribe.assign_blocks(words, [], 0.15) == [0, 0, 0]


def test_assign_blocks_word_gap_is_boundary():
    # word1→word2 のギャップ 0.3 ≥ 0.15 → 境界
    words = [_bw(0.0, 1.0), _bw(1.3, 2.0), _bw(2.0, 3.0)]
    assert transcribe.assign_blocks(words, [], 0.15) == [0, 1, 1]


def test_assign_blocks_silence_maps_to_nearest_joint():
    # 3語連続。無音 [0.95,1.2]（中点1.075）は接合点1.0に最も近い → word1 が境界
    words = [_bw(0.0, 1.0), _bw(1.0, 2.0), _bw(2.0, 3.0)]
    blocks = transcribe.assign_blocks(words, [(0.95, 1.2)], 0.15)
    assert blocks == [0, 1, 1]


def test_assign_blocks_silence_outside_segment_skipped():
    # 無音中点 10.0 はどの単語対の [prev.start, cur.end] にも入らない → 無視
    words = [_bw(0.0, 1.0), _bw(1.0, 2.0)]
    assert transcribe.assign_blocks(words, [(9.9, 10.1)], 0.15) == [0, 0]


def test_assign_blocks_none_times_not_boundary():
    # start/end が None の対はギャップ・無音いずれでも境界にしない
    words = [
        {"word": "a", "start": 0.0, "end": None, "probability": 0.9},
        {"word": "b", "start": None, "end": 2.0, "probability": 0.9},
    ]
    assert transcribe.assign_blocks(words, [(0.5, 1.5)], 0.15) == [0, 0]


def test_assign_blocks_silence_contained_in_word_no_boundary():
    """無音が1単語の span 内部に完全に収まる → 境界は立たない。"""
    # 単語 w0=[0.0,2.0], w1=[2.0,3.0]。無音[0.3,0.8] は w0 に完全包含 → 無視
    words = [_bw(0.0, 2.0), _bw(2.0, 3.0)]
    assert transcribe.assign_blocks(words, [(0.3, 0.8)], 0.15) == [0, 0]


def test_assign_blocks_silence_spanning_words_is_boundary():
    """無音がどの単語にも完全包含されない → 従来どおり境界が立つ。"""
    # w0=[0.0,1.0], w1=[1.0,3.0]。無音[0.8,1.5] は w0 にも w1 にも完全包含されない
    words = [_bw(0.0, 1.0), _bw(1.0, 3.0)]
    blocks = transcribe.assign_blocks(words, [(0.8, 1.5)], 0.15)
    assert blocks == [0, 1]


def test_assign_blocks_silence_start_equals_word_start_is_contained():
    """端一致（s == w.start かつ e < w.end）は包含扱い → 境界なし。"""
    # w0=[0.0,2.0], w1=[2.0,3.0]。無音[0.0,0.5] は s==w0.start かつ e<w0.end
    words = [_bw(0.0, 2.0), _bw(2.0, 3.0)]
    assert transcribe.assign_blocks(words, [(0.0, 0.5)], 0.15) == [0, 0]


# ---- build_segment_line + blocks（／挿入） ----

def test_line_block_separator_inserted():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("結局", 0.9), ("面倒", 0.9))
    line, _ = transcribe.build_segment_line("0001", words, ja, True, blocks=[0, 1])
    assert line == "[0001 0:00] 結局／面倒"


def test_line_block_separator_none_unchanged():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("結局", 0.9), ("面倒", 0.9))
    line, _ = transcribe.build_segment_line("0001", words, ja, True, blocks=None)
    assert line == "[0001 0:00] 結局面倒"


def test_line_block_separator_english_lead_space():
    en = fillers_mod.load_fillers("en")
    words = _words((" hello", 0.9), (" world", 0.9))
    line, _ = transcribe.build_segment_line("0001", words, en, False, blocks=[0, 1])
    body = line.split("] ", 1)[1]
    # ／ は lead 空白の外側（前）に入る
    assert body == " hello／ world"


def test_line_block_separator_outside_brackets_and_before_mark():
    # ／ は 〔〕の外側・◆の前。◆〔〕／ 除去で word 連結に一致（不変条件）
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("動画", 0.3))  # まあ=filler, 動画=low conf
    line, _ = transcribe.build_segment_line("0001", words, ja, True, blocks=[0, 1])
    body = line.split("] ", 1)[1]
    assert body == "〔まあ〕／◆動画"
    cleaned = (
        body.replace(transcribe.LOW_CONF_MARK, "")
        .replace(transcribe.FILLER_OPEN, "")
        .replace(transcribe.FILLER_CLOSE, "")
        .replace(transcribe.BLOCK_SEP, "")
    )
    assert cleaned == "".join(w["word"] for w in words)


def test_line_filler_not_suggested_when_shared_block():
    """同ブロック内のフィラーは巻き込み削除になるため〔〕を付けない。"""
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("どれぐらい", 0.9))
    line, fc = transcribe.build_segment_line("0001", words, ja, True, blocks=[0, 0])
    body = line.split("] ", 1)[1]
    assert "〔" not in body
    assert body == "まあどれぐらい"
    assert fc == 0


def test_line_filler_suggested_when_solo_block():
    """単独ブロックのフィラーは安全に消せるため〔〕を付ける。"""
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("どれぐらい", 0.9))
    line, fc = transcribe.build_segment_line("0001", words, ja, True, blocks=[0, 1])
    body = line.split("] ", 1)[1]
    assert body == "〔まあ〕／どれぐらい"
    assert fc == 1


# ---- suggest_filler_indices（純関数） ----

def test_suggest_solo_block_filler_included():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("どれぐらい", 0.9))
    assert transcribe.suggest_filler_indices(words, [0, 1], ja) == {0}


def test_suggest_shared_block_filler_excluded():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("どれぐらい", 0.9))
    assert transcribe.suggest_filler_indices(words, [0, 0], ja) == set()


def test_suggest_blocks_none_all_fillers():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("まあ", 0.9), ("えー", 0.9), ("結局", 0.9))
    assert transcribe.suggest_filler_indices(words, None, ja) == {0, 1}


def test_suggest_non_filler_excluded():
    ja = fillers_mod.load_fillers("ja")
    words = _words(("結局", 0.9), ("面倒", 0.9))
    assert transcribe.suggest_filler_indices(words, [0, 1], ja) == set()


# ---- transcribe: json の suggest キーと txt の〔〕一致 ----

def test_transcribe_json_suggest_matches_txt(tmp_path, monkeypatch):
    """filler_suggest=True で単独ブロックのフィラー word に suggest=True が付き、
    txt の〔〕位置と一致する。非対象には suggest キーが無い。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "結局まあ面倒",
                "words": [
                    {"word": "結局", "start": 0.0, "end": 0.5, "probability": 0.9},
                    # 0.5→0.8 のギャップ 0.3 ≥ 0.15 → 単独ブロック（前後にポーズ）
                    {"word": "まあ", "start": 0.8, "end": 1.0, "probability": 0.9},
                    {"word": "面倒", "start": 1.3, "end": 1.8, "probability": 0.9},
                ],
            },
        ],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.8)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", filler_suggest=True,
                                pause_threshold=0.15)
    data = res["data"]
    ws = data["segments"][0]["words"]
    assert "suggest" not in ws[0]  # 結局
    assert ws[1].get("suggest") is True  # まあ（単独ブロック）
    assert "suggest" not in ws[2]  # 面倒

    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    assert "〔まあ〕" in txt
    assert res["filler_count"] == 1


def test_transcribe_no_suggest_key_when_disabled(tmp_path, monkeypatch):
    """filler_suggest=False では suggest キーが付かない。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "まあ",
                "words": [
                    {"word": "まあ", "start": 0.8, "end": 1.0, "probability": 0.9},
                ],
            },
        ],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.0)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", filler_suggest=False,
                                pause_threshold=0.15)
    speech = next(
        seg for seg in res["data"]["segments"]
        if seg.get("kind") != "unrecognized"
    )
    ws = speech["words"]
    assert "suggest" not in ws[0]


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


# ---- clamp_words_to_silences（純関数） ----

def _cw(start, end):
    return {"word": "x", "start": start, "end": end, "probability": 0.9}


def test_clamp_end_only_shrinks_to_silence_start():
    # 実測再現: end=2.30 が無音[1.20, 2.35] の内部 → new_end = 1.20
    words = [_cw(1.0, 2.30)]
    out = transcribe.clamp_words_to_silences(words, [(1.20, 2.35)])
    assert out[0]["start"] == pytest.approx(1.0)
    assert out[0]["end"] == pytest.approx(1.20)


def test_clamp_start_only_shrinks_to_silence_end():
    # start=1.5 が無音[1.20, 2.0] の内部 → new_start = 2.0（無音末尾へ）
    words = [_cw(1.5, 3.0)]
    out = transcribe.clamp_words_to_silences(words, [(1.20, 2.0)])
    assert out[0]["start"] == pytest.approx(2.0)
    assert out[0]["end"] == pytest.approx(3.0)


def test_clamp_boundary_exact_no_correction():
    # start==s, end==e は開区間の外 → 補正しない
    words = [_cw(1.0, 2.0)]
    out = transcribe.clamp_words_to_silences(words, [(1.0, 2.0)])
    assert out[0]["start"] == pytest.approx(1.0)
    assert out[0]["end"] == pytest.approx(2.0)


def test_clamp_whole_word_in_one_silence_no_correction():
    # 単語全体が1無音内（無音誤検出の疑い）→ 無補正
    words = [_cw(1.3, 1.8)]
    out = transcribe.clamp_words_to_silences(words, [(1.0, 2.0)])
    assert out[0]["start"] == pytest.approx(1.3)
    assert out[0]["end"] == pytest.approx(1.8)


def test_clamp_both_sides_shrink_below_min_smaller_side_applied():
    # start=1.05（無音[1.0,1.1]内, 補正量0.05）, end=1.12（無音[1.11,1.2]内, 補正量0.01）
    # 両側補正すると new=[1.1, 1.11] 長さ0.01 < 0.05 → 補正量の小さい end 側のみ試す。
    # end 側のみ [1.05, 1.11] 長さ0.06 >= 0.05 → end のみ適用
    words = [_cw(1.05, 1.12)]
    out = transcribe.clamp_words_to_silences(words, [(1.0, 1.1), (1.11, 1.2)])
    assert out[0]["start"] == pytest.approx(1.05)
    assert out[0]["end"] == pytest.approx(1.11)


def test_clamp_both_sides_all_below_min_no_correction():
    # 両側補正・片側補正いずれも min 未満 → 完全無補正
    # start=1.09（無音[1.0,1.1]内）, end=1.111（無音[1.11,1.2]内）
    # both: [1.1,1.11]=0.01<0.05。end のみ:[1.09,1.11]=0.02<0.05。
    # start のみ:[1.1,1.111]=0.011<0.05 → 全滅 → 無補正
    words = [_cw(1.09, 1.111)]
    out = transcribe.clamp_words_to_silences(words, [(1.0, 1.1), (1.11, 1.2)])
    assert out[0]["start"] == pytest.approx(1.09)
    assert out[0]["end"] == pytest.approx(1.111)


def test_clamp_none_times_no_correction():
    words = [
        {"word": "a", "start": None, "end": 2.0, "probability": 0.9},
        {"word": "b", "start": 1.0, "end": None, "probability": 0.9},
    ]
    out = transcribe.clamp_words_to_silences(words, [(0.5, 1.5)])
    assert out[0]["start"] is None and out[0]["end"] == pytest.approx(2.0)
    assert out[1]["start"] == pytest.approx(1.0) and out[1]["end"] is None


def test_clamp_empty_words_and_empty_silences_identity():
    assert transcribe.clamp_words_to_silences([], [(1.0, 2.0)]) == []
    words = [_cw(1.0, 2.0)]
    out = transcribe.clamp_words_to_silences(words, [])
    assert out[0]["start"] == pytest.approx(1.0)
    assert out[0]["end"] == pytest.approx(2.0)


def test_clamp_non_destructive():
    words = [_cw(1.0, 2.30)]
    original = dict(words[0])
    transcribe.clamp_words_to_silences(words, [(1.20, 2.35)])
    assert words[0] == original


# ---- drop_hallucinations（純関数） ----

def _seg(text, words):
    return {"text": text, "words": words}


def _rep_words(word, n):
    return [{"word": word, "start": float(i), "end": float(i) + 0.5,
             "probability": 0.9} for i in range(n)]


def test_drop_halluc_in_segment_repetition():
    # "今、"×10 → セグメント内反復で除去
    seg = _seg("今、" * 10, _rep_words("今、", 10))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_below_top_ratio_kept():
    # 10語中6語同一（0.6 < 0.7）→ 残す
    words = _rep_words("今、", 6) + _rep_words("他", 4)
    seg = _seg("今、今、今、今、今、今、他他他他", words)
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == [seg]
    assert dropped == []


def test_drop_halluc_all_identical_three_removed():
    # 3語以上が全て同一（「今、今、今」型）→ 除去
    seg = _seg("今、" * 3, _rep_words("今、", 3))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_two_identical_kept():
    # 2語同一は除去しない（「はい、はい」等の正常発話がありうる）
    seg = _seg("今、" * 2, _rep_words("今、", 2))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == [seg]
    assert dropped == []


def test_drop_halluc_min_words_boundary_kept():
    # 5語未満で全同一でもない → セグメント内条件では残す
    words = _rep_words("今、", 3) + _rep_words("違う", 1)
    seg = _seg("今、今、今、違う", words)
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == [seg]
    assert dropped == []


def test_drop_halluc_run_three_removed():
    # 同一テキスト3連続 → 3つとも除去
    segs = [_seg("今、", _rep_words("今、", 1)) for _ in range(3)]
    kept, dropped = transcribe.drop_hallucinations(segs)
    assert kept == []
    assert len(dropped) == 3


def test_drop_halluc_run_two_kept():
    # 同一テキスト2連続 → 残す
    segs = [_seg("今、", _rep_words("今、", 1)) for _ in range(2)]
    kept, dropped = transcribe.drop_hallucinations(segs)
    assert kept == segs
    assert dropped == []


def test_drop_halluc_alternating_kept():
    # 交互パターン A/B/A/B → 残す（同一連続ではない）
    segs = [
        _seg("ああして、", _rep_words("ああして、", 1)),
        _seg("こうして、", _rep_words("こうして、", 1)),
        _seg("ああして、", _rep_words("ああして、", 1)),
        _seg("こうして、", _rep_words("こうして、", 1)),
    ]
    kept, dropped = transcribe.drop_hallucinations(segs)
    assert kept == segs
    assert dropped == []


def test_drop_halluc_empty_words_removed():
    seg = _seg("何か", [])
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_empty_text_removed():
    seg = _seg("  ", _rep_words("x", 1))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_punct_only_removed():
    # 記号のみのセグメント（「!」等）→ 除去
    seg = _seg("!", _rep_words("!", 1))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_run_across_dropped_gap():
    # 「今、」が空セグメントを挟んで散在しても連続とみなし RUN=3 で除去
    ima = lambda: _seg("今、", _rep_words("今、", 1))
    empty = lambda: _seg("", [])
    segs = [ima(), empty(), ima(), empty(), ima()]
    kept, dropped = transcribe.drop_hallucinations(segs)
    assert kept == []
    assert len(dropped) == 5


def test_drop_halluc_normal_all_kept():
    segs = [
        _seg("結局", _rep_words("結局", 1)),
        _seg("面倒な作業", _rep_words("面倒", 2)),
        _seg("だった", _rep_words("だった", 1)),
    ]
    kept, dropped = transcribe.drop_hallucinations(segs)
    assert kept == segs
    assert dropped == []


# ---- drop_hallucinations 条件d（時間密度異常）・条件e（エコー） ----

def _timed_words(tokens, start, end):
    """tokens を [start, end] に等間隔で割り付けた words を返す。"""
    n = len(tokens)
    step = (end - start) / n if n else 0.0
    return [
        {"word": t, "start": start + step * i, "end": start + step * (i + 1),
         "probability": 0.9}
        for i, t in enumerate(tokens)
    ]


def test_drop_halluc_density_zero_duration():
    # 5語以上・全単語が同時刻（duration≈0）→ 条件d で除去
    toks = list("あいうえお")
    seg = _seg("あいうえお", _timed_words(toks, 1.0, 1.0))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_density_32chars_012s():
    # 32文字が0.12秒（266文字/秒）の実測パターン → 条件d で除去
    s = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみ"
    assert len(s) == 32
    seg = _seg(s, _timed_words(list(s), 1.0, 1.12))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == []
    assert dropped == [seg]


def test_drop_halluc_density_normal_kept():
    # 36文字4.9秒（約7.3文字/秒）→ 残す
    toks = ["あいうえおか", "きくけこさし", "すせそたちつ",
            "てとなにぬね", "のはひふへほ", "まみむめもや"]
    assert len("".join(toks)) == 36
    seg = _seg("".join(toks), _timed_words(toks, 1.0, 5.9))
    kept, dropped = transcribe.drop_hallucinations([seg])
    assert kept == [seg]
    assert dropped == []


def test_drop_halluc_echo_regression():
    # 実測回帰: A(正常密度・残す), B/C(0.12秒), D(14文字/秒でCのエコー) を除去
    a = "あるいはえー生成AI使ってたの残り使用量みたいなものを見るとえー"
    bc = "えー生成AI使ってたの残り使用量みたいなものを見るとえー"
    d = bc + "どれぐらいトークンを"
    seg_a = _seg(a, _timed_words(list(a), 0.0, 7.6))
    seg_b = _seg(bc, _timed_words(list(bc), 8.0, 8.12))
    seg_c = _seg(bc, _timed_words(list(bc), 8.5, 8.62))
    # D: 密度 = len/duration が 12〜25 の範囲（条件d非該当・条件e該当）
    seg_d = _seg(d, _timed_words(list(d), 9.0, 9.0 + len(d) / 14.0))
    kept, dropped = transcribe.drop_hallucinations(
        [seg_a, seg_b, seg_c, seg_d])
    assert kept == [seg_a]
    assert dropped == [seg_b, seg_c, seg_d]


def test_drop_halluc_echo_skips_empty_intervening_segment():
    # normal / 0.12秒echo / empty / echo+real continuation
    normal_text = "あるいは生成AIの残り使用量を見ると"
    echo_text = "えー生成AIの残り使用量を見ると"
    normal = _seg(normal_text, _timed_words(list(normal_text), 0.0, 5.0))
    short_echo = _seg(echo_text, _timed_words(list(echo_text), 5.0, 5.12))
    empty = _seg("", [])
    extended_text = echo_text + "どれぐらいトークンが残っているか"
    extended_echo = _seg(
        extended_text, _timed_words(list(extended_text), 6.0, 8.0)
    )
    kept, dropped = transcribe.drop_hallucinations(
        [normal, short_echo, empty, extended_echo]
    )
    assert normal in kept
    assert short_echo in dropped
    assert empty in dropped
    assert extended_echo in dropped


def test_drop_halluc_echo_unrelated_high_density_kept():
    # 正常な高速発話でも共通接頭辞がなければ除去しない
    first_text = "今日は会議で新しい計画を詳しく説明します"
    second_text = "明日は顧客と次の提案内容を確認します"
    first = _seg(
        first_text, _timed_words(list(first_text), 0.0, len(first_text) / 15.0)
    )
    second = _seg(
        second_text,
        _timed_words(list(second_text), 2.0, 2.0 + len(second_text) / 15.0),
    )
    kept, dropped = transcribe.drop_hallucinations([first, second])
    assert kept == [first, second]
    assert dropped == []


def test_drop_halluc_echo_ignores_fourth_previous_nonempty_segment():
    # 比較対象は直近3件の非空raw segmentに限定する
    old_text = "生成AIの残り使用量を詳しく確認する"
    old = _seg(old_text, _timed_words(list(old_text), 0.0, 3.0))
    intervening = [
        _seg(text, _timed_words(list(text), start, start + 1.0))
        for text, start in [
            ("最初の別の話題です", 3.0),
            ("続いて別の論点です", 4.0),
            ("最後に異なる話です", 5.0),
        ]
    ]
    current_text = old_text + "追加"
    current = _seg(
        current_text,
        _timed_words(list(current_text), 6.0, 6.0 + len(current_text) / 15.0),
    )
    kept, dropped = transcribe.drop_hallucinations(
        [old, *intervening, current]
    )
    assert kept == [old, *intervening, current]
    assert dropped == []


def test_drop_halluc_echo_normal_density_kept():
    # 直前と共通接頭辞は長いが密度が正常（ゆっくり言い直し）→ 残す
    p = "今日はとても良い天気ですねそうですね"
    q = p + "本当にそう思います"
    seg_p = _seg(p, _timed_words(list(p), 0.0, 4.0))
    seg_q = _seg(q, _timed_words(list(q), 5.0, 11.0))  # 約4.5文字/秒
    kept, dropped = transcribe.drop_hallucinations([seg_p, seg_q])
    assert kept == [seg_p, seg_q]
    assert dropped == []


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
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.8)
    monkeypatch.setattr(transcribe, "_slice_wav", lambda p, start, end: p)
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
        "block": 0,
    }
    # トップレベルに pause_threshold メタが入る
    assert "pause_threshold" in data


def test_transcribe_txt_markers(tmp_path, fake_whisper):
    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", filler_suggest=True)
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    lines = txt.splitlines()
    # seg1: 全語が同ブロック（ギャップなし）→ まあ は単独ブロックでないため〔〕なし
    assert lines[0] == "[0001 0:00] 結局まあ◆面倒"
    # seg2: 最初の単語 start=1.2 → floor=1秒 → 0:01
    assert lines[1] == "[0002 0:01] 作業"
    assert res["filler_count"] == 0


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
    assert kw["temperature"] == transcribe.TEMPERATURE_FALLBACK
    assert kw["condition_on_previous_text"] is False
    assert kw["no_speech_threshold"] == 0.8
    assert kw["compression_ratio_threshold"] == 2.0
    assert (
        kw["hallucination_silence_threshold"]
        == transcribe.HALLUC_SILENCE_SKIP_S
    )


# ---- verbatim モード ----

def test_transcribe_non_verbatim_defaults(tmp_path, fake_whisper):
    """非verbatim: 既定モデル・プロンプトなし・condT False・json verbatim False。"""
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav), lang="ja")
    kw = fake_whisper["kwargs"]
    assert kw["path_or_hf_repo"] == transcribe.DEFAULT_MODEL
    assert kw["condition_on_previous_text"] is False
    assert "initial_prompt" not in kw
    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))
    assert data["verbatim"] is False


def test_transcribe_verbatim_uses_verbatim_model_and_prompt(tmp_path, fake_whisper):
    """verbatim: model未指定→VERBATIM_MODEL・FILLER_PROMPT・condT True・json verbatim True。"""
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav), lang="ja", verbatim=True)
    kw = fake_whisper["kwargs"]
    assert kw["path_or_hf_repo"] == transcribe.VERBATIM_MODEL
    assert kw["initial_prompt"] == transcribe.FILLER_PROMPT
    assert kw["condition_on_previous_text"] is True
    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))
    assert data["verbatim"] is True
    assert data["model"] == transcribe.VERBATIM_MODEL


def test_transcribe_verbatim_disables_fallback_and_silence_skip(tmp_path, fake_whisper):
    """verbatim では温度フォールバックと無音スキップを使わない。

    A/B実測でフィラー候補が67→3〜4に激減する破壊的相互作用があったため
    （温度フォールバック: プロンプトリセットで initial_prompt が消える /
    無音スキップ: 単語異常ヒューリスティクスがフィラーを幻覚と誤判定）。
    """
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav), lang="ja", verbatim=True)
    kw = fake_whisper["kwargs"]
    assert kw["temperature"] == 0
    assert "hallucination_silence_threshold" not in kw


def test_transcribe_verbatim_respects_explicit_model(tmp_path, fake_whisper):
    """verbatimでも明示モデルは尊重する。"""
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav), lang="ja", model="custom-model", verbatim=True)
    assert fake_whisper["kwargs"]["path_or_hf_repo"] == "custom-model"


def test_transcribe_verbatim_en_prompt(tmp_path, fake_whisper):
    """lang=en の verbatim では英語プロンプトに切り替わる。"""
    wav = _make_input(tmp_path)
    transcribe.transcribe(str(wav), lang="en", verbatim=True)
    assert fake_whisper["kwargs"]["initial_prompt"] == transcribe.FILLER_PROMPT_EN


def test_transcribe_passes_runtime_silence_keep_to_build_cuts(
    tmp_path, monkeypatch
):
    """認識用カットは実行時の SILENCE_KEEP_S を明示して組み立てる。"""
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: {"language": "ja", "segments": []}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: "silence_start: 1.0\nsilence_end: 3.0\n",
    )
    monkeypatch.setattr(transcribe, "_write_trimmed_wav", lambda p, cuts: (p, 0.0))
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 3.0)
    monkeypatch.setattr(transcribe, "SILENCE_KEEP_S", 0.25)
    calls = []

    def spy_build_cuts(periods, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(transcribe, "build_cuts", spy_build_cuts)
    transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")
    assert calls == [{"keep": 0.25}]


def test_transcribe_verbatim_chunks_and_offsets_words(tmp_path, monkeypatch):
    calls = []
    clip_paths = []
    slices = []
    silence_calls = []
    planned = {}

    def fake_transcribe(path, **kwargs):
        index = len(calls)
        calls.append((path, kwargs))
        word = (
            {"word": "前", "start": 1.0, "end": 2.0, "probability": 0.9}
            if index == 0
            else {"word": "後", "start": 0.5, "end": 1.5, "probability": 0.9}
        )
        return {
            "language": "ja",
            "segments": [{"text": word["word"], "words": [word]}],
        }

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 600.0)

    def fake_detect(path, d):
        silence_calls.append((path, d))
        if d == 0.5:
            return "silence_start: 299.0\nsilence_end: 301.0\n"
        return ""

    monkeypatch.setattr(transcribe, "_detect_silence", fake_detect)

    def fake_slice(path, start, end):
        slices.append((start, end))
        clip = tmp_path / f"chunk-{len(slices)}.wav"
        clip.write_bytes(b"chunk")
        clip_paths.append(clip)
        return str(clip)

    monkeypatch.setattr(transcribe, "_slice_wav", fake_slice)
    real_plan_chunks = transcribe.plan_chunks

    def spy_plan_chunks(duration, silences, target, tol):
        planned.update(
            duration=duration, silences=silences, target=target, tol=tol
        )
        return real_plan_chunks(duration, silences, target=target, tol=tol)

    monkeypatch.setattr(transcribe, "plan_chunks", spy_plan_chunks)

    wav = _make_input(tmp_path)
    result = transcribe.transcribe(str(wav), lang="ja", verbatim=True)

    assert slices == [(0.0, 300.0), (300.0, 600.0)]
    assert len(calls) == 2
    assert silence_calls[-1][1] == 0.5
    assert planned == {
        "duration": 600.0,
        "silences": [(299.0, 301.0)],
        "target": transcribe.CHUNK_TARGET_S,
        "tol": transcribe.CHUNK_TOL_S,
    }
    for _, kwargs in calls:
        assert kwargs["condition_on_previous_text"] is True
        assert kwargs["temperature"] == 0
        assert kwargs["initial_prompt"] == transcribe.FILLER_PROMPT
    assert [segment.get("kind", "speech") for segment in result["data"]["segments"]] == [
        "unrecognized", "speech", "unrecognized", "speech", "unrecognized"
    ]
    assert (
        result["data"]["segments"][2]["source_start"],
        result["data"]["segments"][2]["source_end"],
    ) == (2.0, 300.5)
    assert result["data"]["segments"][3]["words"][0]["start"] == 300.5
    assert result["data"]["segments"][3]["words"][0]["end"] == 301.5
    assert result["data"]["language"] == "ja"
    assert all(not path.exists() for path in clip_paths)


def test_transcribe_verbatim_deletes_chunk_when_whisper_raises(
    tmp_path, monkeypatch
):
    clip = tmp_path / "failed-chunk.wav"

    def fail_transcribe(path, **kwargs):
        raise RuntimeError("decode failed")

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fail_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 600.0)

    def fake_slice(path, start, end):
        clip.write_bytes(b"chunk")
        return str(clip)

    monkeypatch.setattr(transcribe, "_slice_wav", fake_slice)

    wav = _make_input(tmp_path)
    with pytest.raises(RuntimeError, match="decode failed"):
        transcribe.transcribe(str(wav), lang="ja", verbatim=True)
    assert not clip.exists()


def test_transcribe_non_verbatim_keeps_single_whisper_call(tmp_path, monkeypatch):
    calls = []

    def fake_transcribe(path, **kwargs):
        calls.append((path, kwargs))
        return {"language": "ja", "segments": []}

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(
        transcribe,
        "_wav_duration",
        lambda p: 10.0,
    )
    monkeypatch.setattr(
        transcribe,
        "_slice_wav",
        lambda *args: pytest.fail("non-verbatim must not slice wav"),
    )

    wav = _make_input(tmp_path)
    result = transcribe.transcribe(str(wav), lang="ja", verbatim=False)
    assert len(calls) == 1
    assert result["data"]["segments"] == [
        {
            "id": "0001",
            "kind": "unrecognized",
            "source_start": 0.0,
            "source_end": 10.0,
            "text": "",
            "words": [],
        }
    ]


def test_transcribe_cleans_temp_wavs_when_duration_probe_fails(
    tmp_path, monkeypatch
):
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: pytest.fail(
        "duration failure must happen before ASR"
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    converted = tmp_path / "converted.wav"
    trimmed = tmp_path / "trimmed.wav"
    converted.write_bytes(b"converted")
    trimmed.write_bytes(b"trimmed")
    monkeypatch.setattr(
        transcribe, "_convert_to_16k_mono", lambda p: str(converted)
    )
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(
        transcribe,
        "_write_trimmed_wav",
        lambda p, cuts: (str(trimmed), 0.0),
    )
    monkeypatch.setattr(
        transcribe,
        "_wav_duration",
        lambda p: (_ for _ in ()).throw(RuntimeError("duration failed")),
    )

    with pytest.raises(RuntimeError, match="duration failed"):
        transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert not converted.exists()
    assert not trimmed.exists()


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
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 0.9)
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
        lambda p, **k: "silence_start: 2.0\nsilence_end: 5.0 | silence_duration: 3.0\n",
    )

    res = transcribe.transcribe(str(wav), lang="ja")
    data = res["data"]
    speech = next(
        seg for seg in data["segments"] if seg.get("kind") != "unrecognized"
    )
    w = speech["words"]
    # 「前」rec[1,2] は cut点2.35より前 → そのまま
    assert w[0]["start"] == pytest.approx(1.0)
    assert w[0]["end"] == pytest.approx(2.0)
    # 「後」rec[3,4] は cut点以降 → +2.3
    assert w[1]["start"] == pytest.approx(5.3)
    assert w[1]["end"] == pytest.approx(6.3)
    assert data["silence_trim"]["count"] == 1
    assert data["silence_trim"]["removed_s"] == pytest.approx(2.3, abs=0.01)
    assert res["silence_cut_count"] == 1


def test_transcribe_clamps_word_times_and_saves_silences(tmp_path, monkeypatch):
    """単語 end が無音内部 → json でクランプ済み。silences フィールドに block 無音のみ。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "あの",
                "words": [
                    # end=1.5 が無音[1.20,1.80] の内部 → クランプで end=1.20
                    {"word": "あの", "start": 1.0, "end": 1.5, "probability": 0.9},
                ],
            },
        ],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    # 無音[1.20,1.80]=0.6s（block・clamp対象。1.5s未満なので切り詰めなし）と
    # [3.0,3.05]=0.05s（pause_threshold 未満で block から除外）
    stderr = (
        "silence_start: 1.20\n"
        "silence_end: 1.80 | silence_duration: 0.60\n"
        "silence_start: 3.0\n"
        "silence_end: 3.05 | silence_duration: 0.05\n"
    )
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: stderr)
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.5)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", pause_threshold=0.15)
    data = res["data"]

    speech = next(
        seg for seg in data["segments"] if seg.get("kind") != "unrecognized"
    )
    w = speech["words"][0]
    assert w["start"] == pytest.approx(1.0)
    assert w["end"] == pytest.approx(1.20)  # 無音境界へクランプ

    # silences には pause_threshold(0.15) 以上の無音のみ
    assert data["silences"] == [[1.2, 1.8]]


def test_transcribe_silences_empty_when_pause_threshold_zero(tmp_path, monkeypatch):
    """pause_threshold<=0 では block_silences が空 → silences も空リスト。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "あの",
                "words": [
                    {"word": "あの", "start": 1.0, "end": 1.5, "probability": 0.9},
                ],
            },
        ],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.5)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", pause_threshold=0)
    assert res["data"]["silences"] == []


def test_transcribe_pause_threshold_zero_still_saves_trim_silences(
    tmp_path, monkeypatch
):
    fake_result = {
        "language": "ja",
        "segments": [_speech_segment(0.0, 0.5)],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: "silence_start: 1.0\nsilence_end: 4.0\n",
    )
    monkeypatch.setattr(
        transcribe, "_write_trimmed_wav", lambda p, cuts: (p, 2.3)
    )
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 2.7)

    data = transcribe.transcribe(
        str(_make_input(tmp_path)), lang="ja", pause_threshold=0
    )["data"]

    assert data["silences"] == []
    assert data["trim_silences"] == [[1.0, 4.0]]


def test_transcribe_short_known_silence_at_file_edges_needs_no_warning(
    tmp_path, monkeypatch
):
    fake_result = {
        "language": "ja",
        "segments": [_speech_segment(0.2, 0.8)],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: (
            "silence_start: 0.0\nsilence_end: 0.2\n"
            "silence_start: 0.8\nsilence_end: 1.0\n"
        ),
    )
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.0)

    segments = transcribe.transcribe(
        str(_make_input(tmp_path)), lang="ja"
    )["data"]["segments"]

    assert [seg.get("kind", "speech") for seg in segments] == ["speech"]


def test_transcribe_blocks_and_pause_threshold(tmp_path, monkeypatch):
    """ブロック割当: 単語ギャップ≥閾値で json に block・txt に ／ が入る。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "結局面倒",
                "words": [
                    {"word": "結局", "start": 0.0, "end": 0.5, "probability": 0.9},
                    # 0.5→0.8 のギャップ 0.3 ≥ 0.15 → ブロック境界
                    {"word": "面倒", "start": 0.8, "end": 1.2, "probability": 0.9},
                ],
            },
        ],
    }

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.2)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", pause_threshold=0.15)
    data = res["data"]
    assert data["pause_threshold"] == 0.15
    ws = data["segments"][0]["words"]
    assert ws[0]["block"] == 0
    assert ws[1]["block"] == 1
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    assert txt.splitlines()[0] == "[0001 0:00] 結局／面倒"


def test_transcribe_pause_threshold_zero_no_separator_dense(tmp_path, monkeypatch):
    """pause_threshold=0 は全単語境界（各単語独立ブロック・／は全境界に入る）。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "結局面倒",
                "words": [
                    {"word": "結局", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": "面倒", "start": 0.5, "end": 1.0, "probability": 0.9},
                ],
            },
        ],
    }
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: fake_result
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 1.0)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja", pause_threshold=0)
    ws = res["data"]["segments"][0]["words"]
    assert [w["block"] for w in ws] == [0, 1]
    txt = (tmp_path / "input.sc.txt").read_text(encoding="utf-8")
    assert txt.splitlines()[0] == "[0001 0:00] 結局／面倒"


def test_transcribe_drops_hallucinations_and_renumbers(tmp_path, monkeypatch):
    """幻覚セグメントが json から消え、メタ記録・残セグメントのID詰めを確認。"""
    fake_result = {
        "language": "ja",
        "segments": [
            {
                "text": "結局",
                "words": [
                    {"word": "結局", "start": 0.0, "end": 0.5, "probability": 0.9},
                ],
            },
            # 幻覚: セグメント内反復（"今、"×10）
            {
                "text": "今、" * 10,
                "words": [
                    {"word": "今、", "start": 1.0 + i * 0.1,
                     "end": 1.05 + i * 0.1, "probability": 0.9}
                    for i in range(10)
                ],
            },
            # 幻覚: 空 words
            {"text": "何か", "words": []},
            {
                "text": "面倒",
                "words": [
                    {"word": "面倒", "start": 3.0, "end": 3.5, "probability": 0.9},
                ],
            },
        ],
    }
    # レスキュー再認識でも幻覚しか出ない → 除去にフォールバックする
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "今、" * 10,
                "words": [
                    {"word": "今、", "start": i * 0.1,
                     "end": 0.05 + i * 0.1, "probability": 0.9}
                    for i in range(10)
                ],
            },
        ],
    }

    def fake_transcribe(path, **kwargs):
        if path == "rescue_clip.wav":
            return rescue_result
        return fake_result

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 10.0)
    monkeypatch.setattr(transcribe, "_slice_wav", lambda src, s, e: "rescue_clip.wav")

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    data = res["data"]

    # 幻覚2セグメント（反復＋空words）はレスキューに
    # 失敗しても、未認識区間として保持する
    assert len(data["segments"]) == 4
    assert data["segments"][0]["text"] == "結局"
    assert data["segments"][1]["kind"] == "unrecognized"
    assert data["segments"][2]["text"] == "面倒"
    assert data["segments"][3]["kind"] == "unrecognized"
    assert (
        data["segments"][3]["source_start"],
        data["segments"][3]["source_end"],
    ) == (3.5, 10.0)
    # ID が連番で詰まる
    assert data["segments"][0]["id"] == "0001"
    assert data["segments"][1]["id"] == "0002"
    assert data["segments"][2]["id"] == "0003"
    assert data["segments"][3]["id"] == "0004"
    # メタ記録
    assert res["hallucination_drop_count"] == 2
    assert data["hallucination_drop"]["count"] == 2
    assert len(res["hallucination_ranges"]) >= 1
    # レスキュー失敗 → rescued=False
    assert res["hallucination_ranges"][0]["rescued"] is False


# ---- audio_gap_window / rescue_window（純関数） ----

def test_audio_gap_window_keeps_short_middle_gap():
    assert transcribe.audio_gap_window(5.0, 5.3, 100.0) == (5.0, 5.3)


def test_audio_gap_window_covers_file_edges():
    assert transcribe.audio_gap_window(None, 8.0, 100.0) == (0.0, 8.0)
    assert transcribe.audio_gap_window(92.0, None, 100.0) == (92.0, 100.0)


def test_rescue_window_middle_group():
    # 中間グループ: prev end 〜 next start
    assert transcribe.rescue_window(5.0, 8.0, 100.0) == (5.0, 8.0)


def test_rescue_window_first_group():
    # 先頭グループ: prev なし → 0 〜 next start
    assert transcribe.rescue_window(None, 8.0, 100.0) == (0.0, 8.0)


def test_rescue_window_last_group():
    # 末尾グループ: next なし → prev end 〜 wav長
    assert transcribe.rescue_window(5.0, None, 100.0) == (5.0, 100.0)


def test_rescue_window_full_when_both_none():
    assert transcribe.rescue_window(None, None, 42.0) == (0.0, 42.0)


def test_rescue_window_too_short_returns_none():
    # 窓 0.5 秒未満 → None
    assert transcribe.rescue_window(5.0, 5.3, 100.0) is None


def test_rescue_window_exactly_min_kept():
    # 0.5 秒ちょうどは残す
    assert transcribe.rescue_window(5.0, 5.5, 100.0) == (5.0, 5.5)


# ---- transcribe レスキュー統合（soundfile 系を monkeypatch） ----

def _rescue_main_result():
    """本認識結果: 正常 / 幻覚 / 正常 の3セグメント。"""
    return {
        "language": "ja",
        "segments": [
            {
                "text": "前",
                "words": [
                    {"word": "前", "start": 0.0, "end": 1.0, "probability": 0.9},
                ],
            },
            {
                "text": "今、" * 10,
                "words": [
                    {"word": "今、", "start": 2.0 + i * 0.1,
                     "end": 2.05 + i * 0.1, "probability": 0.9}
                    for i in range(10)
                ],
            },
            {
                "text": "後",
                "words": [
                    {"word": "後", "start": 13.0, "end": 14.0, "probability": 0.9},
                ],
            },
        ],
    }


def _setup_rescue(monkeypatch, main_result, rescue_result, wav_duration=14.0):
    """本認識と rescue で別結果を返す fake を仕込み、呼び出し記録を返す。"""
    calls = []
    rescue_results = (
        list(rescue_result) if isinstance(rescue_result, list) else None
    )

    def fake_transcribe(path, **kwargs):
        calls.append({"path": path, "kwargs": kwargs})
        if len(calls) == 1:
            return main_result
        if rescue_results is None:
            return rescue_result
        return rescue_results.pop(0)

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda p: p)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: wav_duration)
    monkeypatch.setattr(
        transcribe, "_slice_wav", lambda src, s, e: "rescue_clip.wav"
    )
    return calls


def _clean_rescue_result(text="救出", start=0.5, end=1.5):
    return {
        "language": "ja",
        "segments": [
            {
                "text": text,
                "words": [
                    {
                        "word": text,
                        "start": start,
                        "end": end,
                        "probability": 0.9,
                    }
                ],
            }
        ],
    }


def _anomalous_rescue_result():
    return {
        "language": "ja",
        "segments": [
            {
                "text": "今、" * 10,
                "words": [
                    {
                        "word": "今、",
                        "start": i * 0.1,
                        "end": i * 0.1 + 0.05,
                        "probability": 0.9,
                    }
                    for i in range(10)
                ],
            }
        ],
    }


def _echo_rescue_result(suffix):
    common = ["共甲", "共乙", "共丙", "共丁", "共戊", "共己", "共庚", "共辛"]
    tokens = common + [f"末{suffix}", f"尾{suffix}"]
    return {
        "language": "ja",
        "segments": [
            {
                "text": "".join(tokens),
                "words": [
                    {
                        "word": token,
                        "start": i * 0.1,
                        "end": i * 0.1 + 0.09,
                        "probability": 0.9,
                    }
                    for i, token in enumerate(tokens)
                ],
            }
        ],
    }


def test_rescue_uses_verbatim_first_and_accepts_clean_result(
    tmp_path, monkeypatch
):
    calls = _setup_rescue(
        monkeypatch, _rescue_main_result(), _clean_rescue_result()
    )

    result = transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert len(calls) == 2
    kwargs = calls[1]["kwargs"]
    assert kwargs["initial_prompt"] == transcribe.FILLER_PROMPT
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["temperature"] == 0
    assert "hallucination_silence_threshold" not in kwargs
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim"]
    assert rescue_range["status"] == "rescued"


def test_rescue_falls_back_to_safe_mode_after_verbatim_anomaly(
    tmp_path, monkeypatch
):
    calls = _setup_rescue(
        monkeypatch,
        _rescue_main_result(),
        [_anomalous_rescue_result(), _clean_rescue_result()],
    )

    result = transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert len(calls) == 3
    safe_kwargs = calls[2]["kwargs"]
    assert safe_kwargs["condition_on_previous_text"] is False
    assert "initial_prompt" not in safe_kwargs
    assert safe_kwargs["temperature"] == transcribe.TEMPERATURE_FALLBACK
    assert (
        safe_kwargs["hallucination_silence_threshold"]
        == transcribe.HALLUC_SILENCE_SKIP_S
    )
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim", "safe"]
    assert rescue_range["status"] == "rescued"


@pytest.mark.parametrize(
    "verbatim_results",
    [
        [_clean_rescue_result("境界反復") for _ in range(3)],
        [
            _echo_rescue_result("一"),
            _echo_rescue_result("二"),
            _clean_rescue_result("別内容"),
        ],
    ],
    ids=["three-identical-segments", "cross-chunk-echo"],
)
def test_rescue_falls_back_when_combined_verbatim_chunks_are_anomalous(
    tmp_path, monkeypatch, verbatim_results
):
    main_result = _rescue_main_result()
    main_result["segments"][2]["words"][0].update(start=201.0, end=202.0)
    assert all(
        transcribe._valid_rescue(result["segments"])
        for result in verbatim_results
    )
    calls = _setup_rescue(
        monkeypatch,
        main_result,
        [*verbatim_results, _clean_rescue_result("安全復旧")],
    )
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 220.0)

    result = transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert len(calls) == 5
    assert all(
        call["kwargs"]["condition_on_previous_text"] is True
        for call in calls[1:4]
    )
    assert calls[4]["kwargs"]["condition_on_previous_text"] is False
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim", "safe"]
    assert rescue_range["status"] == "rescued"


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        ["not-a-result-dict"],
        {
            "segments": [
                {
                    "text": 123,
                    "words": [
                        {"word": "救出", "start": 0.5, "end": 1.5}
                    ],
                }
            ]
        },
        {
            "segments": [
                {
                    "text": "救出",
                    "words": [
                        {"word": 123, "start": 0.5, "end": 1.5}
                    ],
                }
            ]
        },
    ],
    ids=["result-none", "result-list", "text-non-string", "word-non-string"],
)
def test_invalid_rescue_result_shape_falls_back_then_stays_unrecognized(
    tmp_path, monkeypatch, invalid_result
):
    calls = _setup_rescue(
        monkeypatch,
        _rescue_main_result(),
        [invalid_result, invalid_result],
    )

    result = transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert len(calls) == 3
    unrecognized = result["data"]["segments"][1]
    assert unrecognized["kind"] == "unrecognized"
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim", "safe"]
    assert rescue_range["status"] == "unrecognized"


def test_rescue_failure_returns_unrecognized_segment(tmp_path, monkeypatch):
    calls = _setup_rescue(
        monkeypatch,
        _rescue_main_result(),
        [_anomalous_rescue_result(), _anomalous_rescue_result()],
    )

    result = transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert len(calls) == 3
    unrecognized = result["data"]["segments"][1]
    assert unrecognized["kind"] == "unrecognized"
    assert (unrecognized["source_start"], unrecognized["source_end"]) == (
        1.0,
        13.0,
    )
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim", "safe"]
    assert rescue_range["status"] == "unrecognized"


def test_long_rescue_window_is_split_into_90_second_chunks(
    tmp_path, monkeypatch
):
    calls = []
    slices = []
    clips = []

    def fake_transcribe(path, **kwargs):
        calls.append(kwargs)
        return _clean_rescue_result(text=f"救出{len(calls)}", start=1.0, end=2.0)

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    plan_calls = []
    real_plan_chunks = transcribe.plan_chunks

    def spy_plan_chunks(duration, silences, target, tol):
        plan_calls.append((duration, silences, target, tol))
        return real_plan_chunks(duration, silences, target=target, tol=tol)

    monkeypatch.setattr(transcribe, "plan_chunks", spy_plan_chunks)

    clip_durations = {}

    def fake_slice(src, start, end):
        slices.append((start, end))
        clip = tmp_path / f"rescue-{len(slices)}.wav"
        clip.write_bytes(b"clip")
        clips.append(clip)
        clip_durations[str(clip)] = end - start
        return str(clip)

    monkeypatch.setattr(transcribe, "_slice_wav", fake_slice)
    monkeypatch.setattr(
        transcribe, "_wav_duration", lambda path: clip_durations[path]
    )

    segments = transcribe._rescue_transcribe(
        "source.wav", 0.0, 200.0, "model", "ja", verbatim=True
    )

    assert len(calls) == 3
    assert slices == [(0.0, 90.0), (90.0, 180.0), (180.0, 200.0)]
    assert all(end - start <= 90.0 for start, end in slices)
    assert plan_calls
    assert all(call[2:] == (90.0, 15.0) for call in plan_calls)
    starts = [seg["words"][0]["start"] for seg in segments]
    assert starts == [1.0, 91.0, 181.0]
    assert all(a < b for a, b in zip(starts, starts[1:]))
    assert all(
        kwargs["initial_prompt"] == transcribe.FILLER_PROMPT
        for kwargs in calls
    )
    assert all(not clip.exists() for clip in clips)


def test_verbatim_rescue_falls_back_when_word_is_outside_subchunk(
    tmp_path, monkeypatch
):
    main_result = _rescue_main_result()
    main_result["segments"][2]["words"][0].update(start=201.0, end=202.0)
    outside_chunk = _clean_rescue_result("範囲外", start=100.0, end=101.0)
    calls = _setup_rescue(
        monkeypatch,
        main_result,
        [
            outside_chunk,
            _clean_rescue_result("使われない二番目"),
            _clean_rescue_result("使われない三番目"),
            _clean_rescue_result("安全復旧"),
        ],
    )
    monkeypatch.setattr(
        transcribe,
        "_wav_duration",
        lambda path: 90.0 if path == "rescue_clip.wav" else 220.0,
    )

    result = transcribe.transcribe(str(_make_input(tmp_path)), lang="ja")

    assert len(calls) == 3
    assert calls[1]["kwargs"]["condition_on_previous_text"] is True
    assert calls[2]["kwargs"]["condition_on_previous_text"] is False
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim", "safe"]
    assert rescue_range["status"] == "rescued"


def test_rescue_rejects_words_outside_actual_rounded_clip_duration(
    tmp_path, monkeypatch
):
    """要求0.50003秒でも、16kHz丸め後の0.5秒clip外wordは採用しない。"""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "rounded.wav"
    sf.write(str(wav), np.zeros(8002, dtype="float32"), 16000)
    main_result = _rescue_main_result()
    main_result["segments"][0]["words"][0].update(start=0.0, end=0.0)
    main_result["segments"][2]["words"][0].update(
        start=0.50003, end=0.5001
    )
    rounded_outside = _clean_rescue_result(
        "丸め境界", start=0.50001, end=0.50002
    )
    calls = []

    def fake_transcribe(path, **kwargs):
        calls.append(kwargs)
        return [main_result, rounded_outside, rounded_outside][len(calls) - 1]

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_convert_to_16k_mono", lambda path: path)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda path, **kwargs: "")
    real_slice_wav = transcribe._slice_wav
    slice_requests = []

    def record_slice_wav(src, start, end):
        slice_requests.append((start, end))
        return real_slice_wav(src, start, end)

    monkeypatch.setattr(transcribe, "_slice_wav", record_slice_wav)
    real_wav_duration = transcribe._wav_duration
    clip_durations = []

    def record_wav_duration(path):
        duration = real_wav_duration(path)
        if path != str(wav):
            clip_durations.append(duration)
        return duration

    monkeypatch.setattr(transcribe, "_wav_duration", record_wav_duration)

    result = transcribe.transcribe(str(wav), lang="ja")

    assert len(calls) == 3
    assert slice_requests == [(0.0, 0.50003), (0.0, 0.50003)]
    assert clip_durations == [0.5, 0.5]
    assert calls[1]["condition_on_previous_text"] is True
    assert calls[2]["condition_on_previous_text"] is False
    unrecognized = result["data"]["segments"][1]
    assert unrecognized["kind"] == "unrecognized"
    rescue_range = result["hallucination_ranges"][0]
    assert rescue_range["attempts"] == ["verbatim", "safe"]
    assert rescue_range["status"] == "unrecognized"


@pytest.mark.parametrize(
    ("word_start", "word_end"),
    [(-1.0, 1.0), (89.0, 100.0)],
    ids=["before-clip-start", "after-clip-end"],
)
def test_verbatim_rescue_rejects_partial_word_outside_subchunk_bounds(
    tmp_path, monkeypatch, word_start, word_end
):
    results = [
        _clean_rescue_result("境界", start=word_start, end=word_end),
        _clean_rescue_result("中間"),
        _clean_rescue_result("末尾"),
    ]
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: results.pop(0)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(
        transcribe,
        "_slice_wav",
        lambda src, start, end: str(tmp_path / "missing-clip.wav"),
    )
    monkeypatch.setattr(transcribe, "_wav_duration", lambda path: 90.0)

    segments = transcribe._rescue_transcribe(
        "source.wav", 0.0, 200.0, "model", "ja", verbatim=True
    )

    assert segments == []


def test_long_rescue_prefers_valid_silence_before_90_second_limit(
    tmp_path, monkeypatch
):
    slices = []

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: _clean_rescue_result()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: (
            "silence_start: 84.5\nsilence_end: 85.5\n"
            "silence_start: 97.0\nsilence_end: 101.0\n"
        ),
    )

    def fake_slice(src, start, end):
        slices.append((start, end))
        return str(tmp_path / "missing-clip.wav")

    monkeypatch.setattr(transcribe, "_slice_wav", fake_slice)
    monkeypatch.setattr(transcribe, "_wav_duration", lambda path: 90.0)

    transcribe._rescue_transcribe(
        "source.wav", 0.0, 200.0, "model", "ja", verbatim=True
    )

    assert slices[0] == (0.0, 85.0)


def test_verbatim_rescue_rejects_all_subchunks_when_one_is_anomalous(
    tmp_path, monkeypatch
):
    results = [
        _clean_rescue_result("前救出"),
        _anomalous_rescue_result(),
        _clean_rescue_result("後救出"),
    ]

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda path, **kwargs: results.pop(0)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")
    monkeypatch.setattr(
        transcribe,
        "_slice_wav",
        lambda src, start, end: str(tmp_path / "missing-clip.wav"),
    )
    monkeypatch.setattr(transcribe, "_wav_duration", lambda path: 90.0)

    assert transcribe._rescue_transcribe(
        "source.wav", 0.0, 200.0, "model", "ja", verbatim=True
    ) == []


@pytest.mark.parametrize("failure", [None, "decode", "duration"])
def test_rescue_clip_is_deleted_on_success_and_exception(
    tmp_path, monkeypatch, failure
):
    clip = tmp_path / "rescue-clip.wav"

    def fake_transcribe(path, **kwargs):
        if failure == "decode":
            raise RuntimeError("decode failed")
        return _clean_rescue_result()

    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.setattr(transcribe, "_detect_silence", lambda p, **k: "")

    def fake_slice(src, start, end):
        clip.write_bytes(b"clip")
        return str(clip)

    monkeypatch.setattr(transcribe, "_slice_wav", fake_slice)
    if failure == "duration":
        def fail_duration(path):
            raise RuntimeError("duration failed")

        monkeypatch.setattr(
            transcribe,
            "_wav_duration",
            fail_duration,
        )
    else:
        monkeypatch.setattr(transcribe, "_wav_duration", lambda path: 20.0)

    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            transcribe._rescue_transcribe(
                "source.wav", 0.0, 20.0, "model", "ja", verbatim=True
            )
    else:
        transcribe._rescue_transcribe(
            "source.wav", 0.0, 20.0, "model", "ja", verbatim=True
        )
    assert not clip.exists()


def test_transcribe_rescue_success_inserts_and_offsets(tmp_path, monkeypatch):
    """疎な正常レスキューを残し、前後の未被覆音声も保持する。"""
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "救出テキスト",
                "words": [
                    {"word": "救出", "start": 0.5, "end": 1.5, "probability": 0.9},
                ],
            },
        ],
    }
    calls = _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    data = res["data"]

    assert [s.get("kind", "speech") for s in data["segments"]] == [
        "speech", "unrecognized", "speech", "unrecognized", "speech"
    ]
    assert [s["id"] for s in data["segments"]] == [
        "0001", "0002", "0003", "0004", "0005"
    ]
    assert data["segments"][1]["source_start"] == pytest.approx(1.0)
    assert data["segments"][1]["source_end"] == pytest.approx(1.5)
    assert data["segments"][1]["words"] == []
    assert data["segments"][3]["source_start"] == pytest.approx(2.5)
    assert data["segments"][3]["source_end"] == pytest.approx(13.0)
    assert data["segments"][3]["words"] == []

    # 窓 = 前末尾 end 1.0 〜 後先頭 start 13.0。救出単語 start0.5 に +1.0
    resc = data["segments"][2]["words"][0]
    assert resc["start"] == pytest.approx(1.5)
    assert resc["end"] == pytest.approx(2.5)
    assert (tmp_path / "input.sc.txt").read_text().splitlines() == [
        "[0001 0:00] 前",
        "[0002 0:01] ⚠ 未認識区間 0.5秒（音声保持）",
        "[0003 0:01] 救出",
        "[0004 0:02] ⚠ 未認識区間 10.5秒（音声保持）",
        "[0005 0:13] 後",
    ]

    # whisper は本認識＋レスキューで2回呼ばれる
    assert len(calls) == 2
    # 1段目のレスキューはfresh-context verbatim設定
    rescue_kw = calls[1]["kwargs"]
    assert rescue_kw["condition_on_previous_text"] is True
    assert rescue_kw["initial_prompt"] == transcribe.FILLER_PROMPT

    # メタ: rescued=True・セグメント数
    rng = res["hallucination_ranges"][0]
    assert rng["rescued"] is True
    assert rng["rescued_segments"] == 1


def test_transcribe_rescue_gaps_follow_final_clamped_word_bounds(
    tmp_path, monkeypatch
):
    """未被覆gapを、最終JSONへ出るclamp後word包絡から補完する。"""
    main_result = _rescue_main_result()
    main_result["segments"][2]["words"][0].update(start=6.0, end=7.0)
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "救出テキスト",
                "words": [
                    {
                        "word": "救出",
                        "start": 1.5,
                        "end": 3.5,
                        "probability": 0.9,
                    },
                ],
            },
        ],
    }
    _setup_rescue(
        monkeypatch, main_result, rescue_result, wav_duration=7.0
    )
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: (
            "silence_start: 2.0\nsilence_end: 3.0\n"
            "silence_start: 4.0\nsilence_end: 5.0\n"
        ),
    )

    wav = _make_input(tmp_path)
    segments = transcribe.transcribe(str(wav), lang="ja")["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "speech", "unrecognized", "speech", "unrecognized", "speech"
    ]
    assert (segments[1]["source_start"], segments[1]["source_end"]) == (1.0, 3.0)
    assert [
        (word["start"], word["end"])
        for word in segments[2]["words"]
    ] == [(3.0, 4.0)]
    assert (segments[3]["source_start"], segments[3]["source_end"]) == (4.0, 6.0)


def test_transcribe_rescue_coverage_uses_final_clamped_kept_bounds(
    tmp_path, monkeypatch
):
    """保持窓を前後keptのclamp後境界まで広げ、隙間を残さない。"""
    main_result = _rescue_main_result()
    main_result["segments"][0]["words"][0].update(start=0.0, end=2.5)
    main_result["segments"][2]["words"][0].update(start=4.5, end=6.0)
    _setup_rescue(
        monkeypatch, main_result, {"segments": []}, wav_duration=6.0
    )
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: (
            "silence_start: 2.0\nsilence_end: 3.0\n"
            "silence_start: 4.0\nsilence_end: 5.0\n"
        ),
    )

    wav = _make_input(tmp_path)
    segments = transcribe.transcribe(str(wav), lang="ja")["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "speech", "unrecognized", "speech"
    ]
    assert segments[0]["words"][0]["end"] == 2.0
    assert (segments[1]["source_start"], segments[1]["source_end"]) == (2.0, 5.0)
    assert segments[2]["words"][0]["start"] == 5.0


def test_transcribe_short_untimed_gap_is_preserved_and_reported(
    tmp_path, monkeypatch
):
    """短窓で再認識しなくても、時刻なし除去segmentの保持範囲を記録する。"""
    main_result = {
        "language": "ja",
        "segments": [
            {
                "text": "前",
                "words": [
                    {"word": "前", "start": 4.0, "end": 5.0, "probability": 0.9},
                ],
            },
            {"text": "", "words": []},
            {
                "text": "後",
                "words": [
                    {"word": "後", "start": 5.3, "end": 6.0, "probability": 0.9},
                ],
            },
        ],
    }
    calls = _setup_rescue(
        monkeypatch, main_result, {"segments": []}, wav_duration=6.0
    )

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    u = next(
        seg for seg in res["data"]["segments"]
        if seg.get("kind") == "unrecognized"
        and seg["source_start"] == pytest.approx(5.0)
    )

    assert len(calls) == 1
    assert u["kind"] == "unrecognized"
    assert (u["source_start"], u["source_end"]) == pytest.approx((5.0, 5.3))
    assert res["hallucination_ranges"][0]["start"] == pytest.approx(5.0)
    assert res["hallucination_ranges"][0]["end"] == pytest.approx(5.3)


def test_transcribe_zero_length_gap_does_not_emit_unrecognized(
    tmp_path, monkeypatch
):
    """隣接speechの境界が同時刻なら、長さ0の未認識区間を出力しない。"""
    main_result = {
        "language": "ja",
        "segments": [
            {
                "text": "前",
                "words": [
                    {"word": "前", "start": 4.0, "end": 5.0, "probability": 0.9},
                ],
            },
            {"text": "", "words": []},
            {
                "text": "後",
                "words": [
                    {"word": "後", "start": 5.0, "end": 6.0, "probability": 0.9},
                ],
            },
        ],
    }
    calls = _setup_rescue(
        monkeypatch, main_result, {"segments": []}, wav_duration=6.0
    )

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")

    assert len(calls) == 1
    assert [s.get("kind", "speech") for s in res["data"]["segments"]] == [
        "unrecognized", "speech", "speech"
    ]
    gaps = [
        (s["source_start"], s["source_end"])
        for s in res["data"]["segments"]
        if s.get("kind") == "unrecognized"
    ]
    assert gaps == [(0.0, 4.0)]


def test_transcribe_rescue_rejects_nonmonotonic_words_after_clamping(
    tmp_path, monkeypatch
):
    """clamp後に逆順となるwordは段階レスキューを不合格にする。"""
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "後半救出",
                "words": [
                    {"word": "後半", "start": 10.0, "end": 20.0, "probability": 0.9},
                ],
            },
            {
                "text": "前半救出",
                "words": [
                    {"word": "半", "start": 0.5, "end": 1.0, "probability": 0.9},
                    {"word": "前", "start": -2.0, "end": 0.5, "probability": 0.9},
                ],
            },
        ],
    }
    calls = _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    result = transcribe.transcribe(str(wav), lang="ja")
    segments = result["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "speech", "unrecognized", "speech"
    ]
    assert (
        segments[1]["source_start"], segments[1]["source_end"]
    ) == pytest.approx((1.0, 13.0))
    assert len(calls) == 3
    assert result["hallucination_ranges"][0]["attempts"] == [
        "verbatim", "safe"
    ]


@pytest.mark.parametrize(
    "rescue_segments",
    [
        [
            {
                "text": "長い短い",
                "words": [
                    {"word": "長い", "start": 0.0, "end": 5.0,
                     "probability": 0.9},
                    {"word": "短い", "start": 1.0, "end": 2.0,
                     "probability": 0.9},
                ],
            },
        ],
        [
            {
                "text": "前半救出",
                "words": [
                    {"word": "前半", "start": 0.0, "end": 5.0,
                     "probability": 0.9},
                ],
            },
            {
                "text": "後半救出",
                "words": [
                    {"word": "後半", "start": 4.0, "end": 6.0,
                     "probability": 0.9},
                ],
            },
        ],
    ],
    ids=["nested_words", "overlapping_segments"],
)
def test_transcribe_rescue_with_overlapping_words_falls_back_to_full_gap(
    tmp_path, monkeypatch, rescue_segments
):
    """重複wordを含むレスキューは窓全体を未認識として保持する。"""
    rescue_result = {"language": "ja", "segments": rescue_segments}
    _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    segments = res["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "speech", "unrecognized", "speech"
    ]
    assert (
        segments[1]["source_start"], segments[1]["source_end"]
    ) == pytest.approx((1.0, 13.0))
    assert segments[1]["words"] == []
    assert res["hallucination_ranges"][0]["rescued"] is False
    assert res["hallucination_ranges"][0]["rescued_segments"] == 0


def test_transcribe_rescue_with_untimed_segment_falls_back_to_full_gap(
    tmp_path, monkeypatch
):
    """時刻なしsegmentを含むレスキューは成功扱いせず窓全体を保持する。"""
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "救出テキスト",
                "words": [
                    {"word": "救出", "start": 0.5, "end": 1.5, "probability": 0.9},
                ],
            },
            {
                "text": "時刻なし",
                "words": [
                    {"word": "時刻なし", "start": None, "end": None,
                     "probability": 0.9},
                ],
            },
        ],
    }
    _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    segments = res["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "speech", "unrecognized", "speech"
    ]
    assert (
        segments[1]["source_start"], segments[1]["source_end"]
    ) == pytest.approx((1.0, 13.0))
    assert res["hallucination_ranges"][0]["rescued"] is False
    assert res["hallucination_ranges"][0]["rescued_segments"] == 0


def test_transcribe_rescue_with_invalid_times_falls_back_to_full_gap(
    tmp_path, monkeypatch
):
    """正規化不能な時刻を含むレスキューは例外化せず窓全体を保持する。"""
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "異常時刻のレスキュー結果です",
                "words": [
                    {
                        "word": f"語{i}",
                        "start": "invalid" if i == 0 else i * 0.1,
                        "end": i * 0.1 + 0.05,
                        "probability": 0.9,
                    }
                    for i in range(10)
                ],
            },
        ],
    }
    _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    segments = res["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "speech", "unrecognized", "speech"
    ]
    assert (
        segments[1]["source_start"], segments[1]["source_end"]
    ) == pytest.approx((1.0, 13.0))


def test_transcribe_leading_and_trailing_gaps_cover_file_edges(
    tmp_path, monkeypatch
):
    """先頭・末尾のレスキュー失敗は保持窓をファイル端まで広げる。"""
    hallucination = {
        "text": "今、" * 10,
        "words": [
            {
                "word": "今、",
                "start": 1.0 + i * 0.1,
                "end": 1.05 + i * 0.1,
                "probability": 0.9,
            }
            for i in range(10)
        ],
    }
    main_result = {
        "language": "ja",
        "segments": [
            hallucination,
            {
                "text": "中",
                "words": [
                    {"word": "中", "start": 8.0, "end": 9.0, "probability": 0.9},
                ],
            },
            {
                **hallucination,
                "words": [
                    {**word, "start": word["start"] + 9.0, "end": word["end"] + 9.0}
                    for word in hallucination["words"]
                ],
            },
        ],
    }
    _setup_rescue(
        monkeypatch, main_result, {"segments": []}, wav_duration=20.0
    )

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    segments = res["data"]["segments"]

    assert [s.get("kind", "speech") for s in segments] == [
        "unrecognized", "speech", "unrecognized"
    ]
    assert (segments[0]["source_start"], segments[0]["source_end"]) == (0.0, 8.0)
    assert (segments[2]["source_start"], segments[2]["source_end"]) == (9.0, 20.0)
    assert [
        (r["start"], r["end"]) for r in res["hallucination_ranges"]
    ] == [(0.0, 8.0), (9.0, 20.0)]


def test_transcribe_sparse_rescue_gaps_are_remapped_once_with_cuts(
    tmp_path, monkeypatch
):
    """未被覆gapは元音源時刻へ1回だけ変換し、最終出力で再変換しない。"""
    main_result = _rescue_main_result()
    main_result["segments"][2]["words"][0].update(start=7.0, end=8.0)
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "救出テキスト",
                "words": [
                    {"word": "救出", "start": 2.0, "end": 3.0, "probability": 0.9},
                ],
            },
        ],
    }
    _setup_rescue(monkeypatch, main_result, rescue_result)
    monkeypatch.setattr(
        transcribe,
        "_detect_silence",
        lambda p, **k: "silence_start: 2.0\nsilence_end: 5.0\n",
    )
    monkeypatch.setattr(
        transcribe, "_write_trimmed_wav", lambda path, cuts: (path, 2.3)
    )
    monkeypatch.setattr(transcribe, "_wav_duration", lambda p: 10.0)

    wav = _make_input(tmp_path)
    segments = transcribe.transcribe(str(wav), lang="ja")["data"]["segments"]

    assert (segments[1]["source_start"], segments[1]["source_end"]) == pytest.approx(
        (1.0, 5.3)
    )
    assert (segments[2]["words"][0]["start"], segments[2]["words"][0]["end"]) == pytest.approx(
        (5.3, 6.3)
    )
    assert (segments[3]["source_start"], segments[3]["source_end"]) == pytest.approx(
        (6.3, 9.3)
    )


def test_transcribe_rescue_fallback_when_partly_hallucinated(
    tmp_path, monkeypatch
):
    """レスキュー結果の一部が幻覚なら窓全体を未認識として保持する。"""
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "救出テキスト",
                "words": [
                    {
                        "word": "救出",
                        "start": 0.5,
                        "end": 1.5,
                        "probability": 0.9,
                    },
                ],
            },
            {
                "text": "今、" * 10,
                "words": [
                    {
                        "word": "今、",
                        "start": 2.0 + i * 0.1,
                        "end": 2.05 + i * 0.1,
                        "probability": 0.9,
                    }
                    for i in range(10)
                ],
            },
        ],
    }
    _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    data = res["data"]

    assert [s.get("kind", "speech") for s in data["segments"]] == [
        "speech", "unrecognized", "speech"
    ]
    u = data["segments"][1]
    assert u["source_start"] == pytest.approx(1.0)
    assert u["source_end"] == pytest.approx(13.0)
    assert u["words"] == []
    rng = res["hallucination_ranges"][0]
    assert rng["rescued"] is False
    assert rng["rescued_segments"] == 0


def test_transcribe_rescue_fallback_when_hallucination(tmp_path, monkeypatch):
    """レスキュー結果も幻覚なら未認識区間として保持する。"""
    rescue_result = {
        "language": "ja",
        "segments": [
            {
                "text": "今、" * 10,
                "words": [
                    {"word": "今、", "start": i * 0.1,
                     "end": 0.05 + i * 0.1, "probability": 0.9}
                    for i in range(10)
                ],
            },
        ],
    }
    _setup_rescue(monkeypatch, _rescue_main_result(), rescue_result)

    wav = _make_input(tmp_path)
    res = transcribe.transcribe(str(wav), lang="ja")
    data = res["data"]

    assert [s.get("kind", "speech") for s in data["segments"]] == [
        "speech", "unrecognized", "speech"
    ]
    u = data["segments"][1]
    assert u["source_start"] == pytest.approx(1.0)
    assert u["source_end"] == pytest.approx(13.0)
    assert u["words"] == []
    assert "未認識区間" in (tmp_path / "input.sc.txt").read_text()
    rng = res["hallucination_ranges"][0]
    assert rng["rescued"] is False
    assert rng["rescued_segments"] == 0


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
    # 全語が同ブロック（ギャップなし）→ um は単独ブロックでないため〔〕なし
    assert lines[0] == "[0001 0:00]  so um yeah"
    assert res["filler_count"] == 0
