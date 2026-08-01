import base64
import http.client
import json
import os
import threading

import numpy as np
import pytest

from stefnceorf import render, serve


# ---- テスト用プロジェクト（合成wav + 手書き json） ----

def _make_project(tmp_path, sr=16000):
    """Hlyst 検証用の最小プロジェクト。

    0001 speech: あ[0,1)220 / えー[1.1,1.6)440(suggest, 前後無音→安全カット) /
                 い[1.8,2.8)660
    0002 unrecognized(uts): [2.8,3.8)（島は[2.9,3.7)。前後無音→安全カット）
    0003 speech: う[3.8,4.8)880
    0004 speech: そう[7.0,7.5) / あの[7.5,8.0)(suggest, 前後が発話と連続→unsafe) /
                 ね[8.0,8.5)
    """
    import soundfile as sf

    total_s = 9.0
    audio = np.zeros(int(round(total_s * sr)))

    def _tone(start, end, freq):
        t = np.arange(int(round((end - start) * sr))) / sr
        a = int(round(start * sr))
        audio[a: a + len(t)] = 0.3 * np.sin(2 * np.pi * freq * t)

    _tone(0.0, 1.0, 220.0)
    _tone(1.1, 1.6, 440.0)
    _tone(1.8, 2.8, 660.0)
    _tone(2.9, 3.7, 550.0)  # UTS 島（前後に無音の縁があり安全カットできる）
    _tone(3.8, 4.8, 880.0)
    _tone(7.0, 8.5, 330.0)  # 0004 は3語が連続（無音境界なし）

    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "pause_threshold": 0.15,
        "silences": [[1.0, 1.1], [1.6, 1.8], [2.8, 2.9], [3.7, 3.8], [4.8, 7.0]],
        "trim_silences": [[4.8, 7.0]],
        "segments": [
            {
                "id": "0001",
                "text": "あえーい",
                "words": [
                    {"word": "あ", "start": 0.0, "end": 1.0,
                     "probability": 0.9, "block": 0},
                    {"word": "えー", "start": 1.1, "end": 1.6,
                     "probability": 0.9, "block": 1, "suggest": True},
                    {"word": "い", "start": 1.8, "end": 2.8,
                     "probability": 0.9, "block": 2},
                ],
            },
            {
                "id": "0002",
                "kind": "unrecognized",
                "reason": "uts",
                "source_start": 2.8,
                "source_end": 3.8,
                "text": "",
                "words": [],
            },
            {
                "id": "0003",
                "text": "う",
                "words": [
                    {"word": "う", "start": 3.8, "end": 4.8,
                     "probability": 0.9, "block": 3},
                ],
            },
            {
                "id": "0004",
                "text": "そうあのね",
                "words": [
                    {"word": "そう", "start": 7.0, "end": 7.5,
                     "probability": 0.9, "block": 4},
                    {"word": "あの", "start": 7.5, "end": 8.0,
                     "probability": 0.9, "block": 4, "suggest": True},
                    {"word": "ね", "start": 8.0, "end": 8.5,
                     "probability": 0.9, "block": 4},
                ],
            },
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


_TXT = (
    "[0001 0:00] あ〔えー〕い\n"
    "[0002 0:02] 〔未転写発話 1.0秒〕\n"
    "[0003 0:03] う\n"
    "[0004 0:07] そう〔あの〕ね\n"
)


def _write_txt(tmp_path, body=_TXT):
    txt = tmp_path / "input.sc.txt"
    txt.write_text(body, encoding="utf-8")
    return txt


def _session(tmp_path, body=_TXT):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, body)
    return serve.HlystSession(str(txt)), txt


def _data(tmp_path):
    return json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))


def _fillers():
    from stefnceorf import fillers as fillers_mod

    return fillers_mod.load_fillers("ja")


def _apply(tmp_path, text, toggles):
    return serve.apply_toggles(text, _data(tmp_path), _fillers(), toggles)


def _seg(segments, seg_id):
    return next(s for s in segments if s["id"] == seg_id)


# ---- raw 行のパース補助 ----

def test_split_eol_variants():
    assert serve.split_eol("a\r\n") == ("a", "\r\n")
    assert serve.split_eol("a\n") == ("a", "\n")
    assert serve.split_eol("a") == ("a", "")


def test_line_segment_id_and_body():
    line = "[0001 0:00] あ〔えー〕い\n"
    assert serve.line_segment_id(line) == "0001"
    assert serve.line_body(line) == "あ〔えー〕い"


def test_line_segment_id_ignores_non_id_line():
    assert serve.line_segment_id("メモ行\n") is None
    assert serve.line_body("メモ行\n") == ""


def test_detect_newline_prefers_crlf():
    assert serve.detect_newline("a\r\nb\r\n") == "\r\n"
    assert serve.detect_newline("a\nb\n") == "\n"


# ---- 単語の構造照合（raw 文字範囲の特定） ----

def _words(tmp_path, seg_id="0001"):
    return _seg(_data(tmp_path)["segments"], seg_id)["words"]


def test_locate_word_span_plain(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    assert serve.locate_word_span("あえーい", words, 1) == (1, 3)


def test_locate_word_span_with_brackets_and_markers(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    body = "あ／◆〔えー〕い"
    start, end = serve.locate_word_span(body, words, 1)
    assert body[start:end] == "えー"


def test_locate_word_span_survives_hand_edit(tmp_path):
    """句読点を手で足した行でも本体文字は一意に対応づけられる。"""
    _make_project(tmp_path)
    words = _words(tmp_path)
    body = "あ、えー、い。"
    start, end = serve.locate_word_span(body, words, 1)
    assert body[start:end] == "えー"


def test_locate_word_span_returns_none_when_word_deleted(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    assert serve.locate_word_span("あい", words, 1) is None


def test_locate_word_span_picks_correct_duplicate_occurrence(tmp_path):
    """同じフィラーが複数出現しても word_index に対応する方を選ぶ。"""
    _make_project(tmp_path)
    words = [
        {"word": "えー", "start": 0.0, "end": 0.5, "suggest": True},
        {"word": "あ", "start": 0.5, "end": 1.0},
        {"word": "えー", "start": 1.0, "end": 1.5, "suggest": True},
    ]
    body = "えーあえー"
    assert serve.locate_word_span(body, words, 0) == (0, 2)
    assert serve.locate_word_span(body, words, 2) == (3, 5)


def test_bracket_word_spans_maps_duplicates_in_order(tmp_path):
    words = [
        {"word": "えー", "start": 0.0, "end": 0.5, "suggest": True},
        {"word": "あ", "start": 0.5, "end": 1.0},
        {"word": "えー", "start": 1.0, "end": 1.5, "suggest": True},
    ]
    spans = serve.bracket_word_spans("えーあ〔えー〕", words)
    assert list(spans) == [2]


# ---- 〔〕の付与 / 解除 ----

def test_set_word_bracket_adds(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    assert serve.set_word_bracket("あえーい", words, 1, True) == "あ〔えー〕い"


def test_set_word_bracket_removes(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    assert serve.set_word_bracket("あ〔えー〕い", words, 1, False) == "あえーい"


def test_set_word_bracket_noop_keeps_body_identical(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    body = "あ、〔えー〕い。"
    assert serve.set_word_bracket(body, words, 1, True) is body


def test_set_word_bracket_keeps_low_conf_mark_outside(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    out = serve.set_word_bracket("あ◆えーい", words, 1, True)
    assert out == "あ◆〔えー〕い"


def test_set_word_bracket_error_when_word_deleted(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    with pytest.raises(serve.TxtEditError) as exc:
        serve.set_word_bracket("あい", words, 1, True)
    assert "直接編集" in str(exc.value)


def test_set_word_bracket_error_when_restoring_deleted_word(tmp_path):
    _make_project(tmp_path)
    words = _words(tmp_path)
    with pytest.raises(serve.TxtEditError):
        serve.set_word_bracket("あい", words, 1, False)


def test_set_word_bracket_error_on_ambiguous_duplicate_after_manual_delete():
    """同一テキストの語が複数あり1語が手動削除済みだと、残存全語のトグルが失敗。

    json「え／え／え」・raw「ええ」（1語を手動削除済み）→ SequenceMatcher の
    整列は恣意的なので、どの「え」に対する操作も TxtEditError にする。
    """
    words = [
        {"word": "え", "start": 0.0, "end": 0.3},
        {"word": "え", "start": 0.3, "end": 0.6},
        {"word": "え", "start": 0.6, "end": 0.9},
    ]
    body = "ええ"
    for idx in range(3):
        with pytest.raises(serve.TxtEditError):
            serve.set_word_bracket(body, words, idx, True)


def test_set_word_bracket_succeeds_when_all_duplicates_uniquely_mapped():
    """同一テキストの語が複数あっても全語が raw へ一意対応できれば従来どおり成功する。"""
    words = [
        {"word": "え", "start": 0.0, "end": 0.3},
        {"word": "え", "start": 0.3, "end": 0.6},
    ]
    body = "ええ"
    assert serve.set_word_bracket(body, words, 0, True) == "〔え〕え"
    assert serve.set_word_bracket(body, words, 1, True) == "え〔え〕"


def test_set_line_bracket_round_trip(tmp_path):
    _make_project(tmp_path)
    seg = _seg(_data(tmp_path)["segments"], "0002")
    off = serve.set_line_bracket("〔未転写発話 1.0秒〕", seg, False)
    assert off == "未転写発話 1.0秒"
    assert serve.set_line_bracket(off, seg, True) == "〔未転写発話 1.0秒〕"


# ---- apply_toggles（raw 行の局所適用） ----

def _all_toggles(tmp_path, text):
    """現在の状態と同じトグル列（＝何も変えない指定）を作る。"""
    state = serve.requested_state(text, _data(tmp_path), _fillers())
    toggles = [
        {"type": "line", "id": sid, "delete": val}
        for sid, val in state["segments"].items()
    ]
    for sid, words in state["words"].items():
        for wi, val in words.items():
            toggles.append({"type": "word", "id": sid, "word": wi, "delete": val})
    return toggles


def test_apply_toggles_noop_keeps_bytes_identical(tmp_path):
    """現状と同じ指定なら1バイトも変わらない（手編集を壊さない）。"""
    _make_project(tmp_path)
    text = (
        "[0001 0:00] あ、〔えー〕い。\n"
        "[0002 0:02] 〔未転写発話 1.0秒〕\n"
        "\n"
        "[0003 0:03] う！\n"
        "[0004 0:07] そう〔あの〕ね\n"
    )
    assert _apply(tmp_path, text, _all_toggles(tmp_path, text)) == text


def test_apply_toggles_changes_only_target_line(tmp_path):
    """1件のトグルで対象行だけが変わり、手編集済みの他行はバイト不変。"""
    _make_project(tmp_path)
    text = (
        "[0001 0:00] あ、〔えー〕い。\n"
        "[0002 0:02] 〔未転写発話 1.0秒〕\n"
        "[0003 0:03] う！！\n"
        "[0004 0:07] そう〔あの〕ね\n"
    )
    out = _apply(
        tmp_path, text,
        [{"type": "word", "id": "0001", "word": 1, "delete": False}],
    )
    before = text.splitlines(keepends=True)
    after = out.splitlines(keepends=True)
    assert after[0] == "[0001 0:00] あ、えーい。\n"
    assert after[1:] == before[1:]


def test_apply_toggles_deletes_line(tmp_path):
    _make_project(tmp_path)
    out = _apply(tmp_path, _TXT, [{"type": "line", "id": "0003", "delete": True}])
    assert "[0003" not in out
    assert out.splitlines(keepends=True) == [
        line for line in _TXT.splitlines(keepends=True) if "[0003" not in line
    ]


def test_apply_toggles_restores_line_in_json_order(tmp_path):
    """削除済み行の復活は json から標準形式で再生成し、json 順の位置へ挿入する。"""
    _make_project(tmp_path)
    text = "[0001 0:00] あ〔えー〕い\n[0004 0:07] そう〔あの〕ね\n"
    out = _apply(tmp_path, text, [{"type": "line", "id": "0003", "delete": False}])
    lines = out.splitlines()
    assert [serve.line_segment_id(x) for x in lines] == ["0001", "0003", "0004"]
    assert lines[1] == "[0003 0:03] う"


def test_apply_toggles_restore_appends_when_no_following_line(tmp_path):
    _make_project(tmp_path)
    text = "[0001 0:00] あ〔えー〕い\n"
    out = _apply(tmp_path, text, [{"type": "line", "id": "0004", "delete": False}])
    assert out.splitlines()[-1] == "[0004 0:07] そう〔あの〕ね"


def test_apply_toggles_restore_adds_missing_final_newline(tmp_path):
    _make_project(tmp_path)
    text = "[0001 0:00] あ〔えー〕い"
    out = _apply(tmp_path, text, [{"type": "line", "id": "0003", "delete": False}])
    assert out == "[0001 0:00] あ〔えー〕い\n[0003 0:03] う\n"


def test_apply_toggles_keeps_crlf(tmp_path):
    _make_project(tmp_path)
    text = _TXT.replace("\n", "\r\n")
    out = _apply(
        tmp_path, text, [{"type": "word", "id": "0001", "word": 1, "delete": False}]
    )
    assert out.splitlines(keepends=True)[0] == "[0001 0:00] あえーい\r\n"
    assert out.count("\r\n") == 4


def test_apply_toggles_uts_bracket_off_keeps_line(tmp_path):
    _make_project(tmp_path)
    out = _apply(tmp_path, _TXT, [{"type": "line", "id": "0002", "delete": False}])
    assert "[0002 0:02] 未転写発話 1.0秒\n" in out


def test_apply_toggles_uts_bracket_round_trip(tmp_path):
    _make_project(tmp_path)
    off = _apply(tmp_path, _TXT, [{"type": "line", "id": "0002", "delete": False}])
    on = _apply(tmp_path, off, [{"type": "line", "id": "0002", "delete": True}])
    assert on == _TXT


def test_apply_toggles_word_toggle_ignored_for_deleted_line(tmp_path):
    _make_project(tmp_path)
    text = "[0002 0:02] 〔未転写発話 1.0秒〕\n[0003 0:03] う\n"
    out = _apply(
        tmp_path, text, [{"type": "word", "id": "0001", "word": 1, "delete": False}]
    )
    assert out == text


def test_apply_toggles_error_when_word_manually_deleted(tmp_path):
    _make_project(tmp_path)
    text = "[0001 0:00] あい\n"
    with pytest.raises(serve.TxtEditError):
        _apply(
            tmp_path, text,
            [{"type": "word", "id": "0001", "word": 1, "delete": False}],
        )


# ---- requested_state ----

def test_requested_state_reflects_brackets_and_missing_lines(tmp_path):
    _make_project(tmp_path)
    text = "[0001 0:00] あ〔えー〕い\n[0004 0:07] そうあのね\n"
    state = serve.requested_state(text, _data(tmp_path), _fillers())
    assert state["segments"] == {
        "0001": False, "0002": True, "0003": True, "0004": False
    }
    assert state["words"]["0001"][1] is True
    assert state["words"]["0004"][1] is False


def test_requested_state_marks_manually_deleted_word(tmp_path):
    _make_project(tmp_path)
    state = serve.requested_state(
        "[0001 0:00] あい\n", _data(tmp_path), _fillers()
    )
    assert state["words"]["0001"][1] is True


# ---- 状態モデル（requested / effective の分離） ----

def test_unsafe_filler_is_requested_but_not_effective(tmp_path):
    """〔〕あり＋unsafe → requested=True / effective=False。"""
    session, _txt = _session(tmp_path)
    data = session.review_data(session.raw_text, session.plan)
    word = _seg(data["segments"], "0004")["words"][1]
    assert word["requested_delete"] is True
    assert word["effective_delete"] is False
    safe = _seg(data["segments"], "0001")["words"][1]
    assert (safe["requested_delete"], safe["effective_delete"]) == (True, True)


def test_saving_other_item_keeps_unsafe_brackets(tmp_path):
    """unsafe 項目を含む状態で別項目を保存しても〔〕が勝手に外れない。"""
    session, txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True
    session.save(toggles)
    body = txt.read_text(encoding="utf-8")
    assert "そう〔あの〕ね" in body
    assert "[0003" not in body


def test_unsafe_toggle_off_then_on_matches_raw_txt(tmp_path):
    """unsafe 項目の ON/OFF 後に raw TXT と表示状態が一致する。"""
    session, txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)

    def _set(word_delete):
        for t in toggles:
            if t["type"] == "word" and t["id"] == "0004" and t["word"] == 1:
                t["delete"] = word_delete

    _set(False)
    result = session.save(list(toggles))
    assert "そうあのね" in txt.read_text(encoding="utf-8")
    assert _seg(result["segments"], "0004")["words"][1]["requested_delete"] is False

    _set(True)
    result = session.save(list(toggles))
    assert "そう〔あの〕ね" in txt.read_text(encoding="utf-8")
    word = _seg(result["segments"], "0004")["words"][1]
    assert word["requested_delete"] is True
    assert word["effective_delete"] is False


def test_preview_center_of_unsafe_word_is_word_center(tmp_path):
    """effective=False（unsafe で保持）の preview 中心は対象中央になる。"""
    session, _txt = _session(tmp_path)
    result = session.preview(
        _all_toggles(tmp_path, session.raw_text),
        {"type": "word", "id": "0004", "word": 1},
    )
    # 0004 の前に無音トリムが入るため出力時刻は原音より前だが、
    # 単語中央 7.75s の写像であること（結合境界＝行頭ではない）を確認する
    data = session.review_data(session.raw_text, session.plan)
    seg = _seg(data["segments"], "0004")
    assert result["center"] > seg["output"]
    assert "安全判定により保持" in result["message"]


def test_preview_center_of_cut_word_is_join_boundary(tmp_path):
    """削除ON（安全カット済み）の preview 中心は結合境界になる。"""
    session, _txt = _session(tmp_path)
    result = session.preview(
        _all_toggles(tmp_path, session.raw_text),
        {"type": "word", "id": "0001", "word": 1},
    )
    # えー[1.1,1.6) はカット済み。中心は「あ」の終端＝結合境界の近傍
    assert 0.9 < result["center"] < 1.3
    assert result["message"] == "現在案プレビュー：削除"


def test_preview_does_not_write_txt(tmp_path):
    session, txt = _session(tmp_path)
    before = txt.read_bytes()
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "word" and t["id"] == "0001" and t["word"] == 1:
            t["delete"] = False
    result = session.preview(toggles, {"type": "word", "id": "0001", "word": 1})
    assert txt.read_bytes() == before
    word = _seg(result["segments"], "0001")["words"][1]
    assert word["requested_delete"] is False
    assert word["effective_delete"] is False


def test_preview_on_off_differ_in_audio_and_message(tmp_path):
    """削除ON/OFFで試聴音声のバイト列とメッセージが変わる。"""
    session, _txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)

    def _set(delete):
        for t in toggles:
            if t["type"] == "line" and t["id"] == "0003":
                t["delete"] = delete

    _set(True)
    on_result = session.preview(list(toggles), {"type": "line", "id": "0003"})
    _set(False)
    off_result = session.preview(list(toggles), {"type": "line", "id": "0003"})

    assert on_result["message"] == "現在案プレビュー：削除"
    assert off_result["message"] == "現在案プレビュー：保持"
    assert on_result["audio"] != off_result["audio"]


def test_preview_returns_playable_wav(tmp_path):
    import io

    import soundfile as sf

    session, _txt = _session(tmp_path)
    result = session.preview(
        _all_toggles(tmp_path, session.raw_text), {"type": "line", "id": "0003"}
    )
    audio, sr = sf.read(io.BytesIO(base64.b64decode(result["audio"])))
    assert sr == 16000
    assert 0 < len(audio) <= int(2 * serve.PREVIEW_PAD_DEFAULT_S * sr)


def test_preview_unknown_segment_raises(tmp_path):
    session, _txt = _session(tmp_path)
    with pytest.raises(ValueError):
        session.preview([], {"type": "line", "id": "9999"})


# ---- preview 範囲（pad_s）と絶対時刻 ----

@pytest.mark.parametrize("pad", [2.0, 4.0, 8.0])
def test_preview_window_length_matches_pad(tmp_path, pad):
    import io

    import soundfile as sf

    session, _txt = _session(tmp_path)
    result = session.preview(
        _all_toggles(tmp_path, session.raw_text),
        {"type": "line", "id": "0003"},
        pad_s=pad,
    )
    audio, sr = sf.read(io.BytesIO(base64.b64decode(result["audio"])))
    assert 0 < len(audio) <= int(2 * pad * sr)
    assert result["pad_s"] == pad


def test_preview_window_clamped_at_boundaries(tmp_path):
    """先頭付近の対象は pad_s ぶんフルには取れず、窓長が短くなる（境界クランプ）。"""
    session, _txt = _session(tmp_path)
    result = session.preview(
        _all_toggles(tmp_path, session.raw_text),
        {"type": "line", "id": "0001"},
        pad_s=8.0,
    )
    assert result["window_start_s"] == 0.0
    assert result["window_end_s"] - result["window_start_s"] < 2 * 8.0


def test_preview_response_includes_absolute_times(tmp_path):
    session, _txt = _session(tmp_path)
    result = session.preview(
        _all_toggles(tmp_path, session.raw_text),
        {"type": "line", "id": "0003"},
    )
    assert result["target_source_s"] > 0
    assert result["center_s"] == result["center"]
    assert result["window_start_s"] <= result["center_s"] <= result["window_end_s"]
    assert result["window_end_s"] > result["window_start_s"]


def test_validate_pad_s_allows_known_values_and_default():
    assert serve._validate_pad_s(None) == serve.PREVIEW_PAD_DEFAULT_S
    assert serve._validate_pad_s(2) == 2.0
    assert serve._validate_pad_s(4.0) == 4.0
    assert serve._validate_pad_s(8) == 8.0


@pytest.mark.parametrize("bad", [3, 0, "4", True, 4.5])
def test_validate_pad_s_rejects_unknown_values(bad):
    with pytest.raises(ValueError):
        serve._validate_pad_s(bad)


def test_preview_endpoint_rejects_invalid_pad_s(server):
    httpd, _url, _txt = server
    status, _h, body = _post(
        httpd, "/preview",
        {
            "toggles": [],
            "target": {"type": "line", "id": "0003"},
            "pad_s": 3,
        },
    )
    assert status == 400
    assert "pad_s" in json.loads(body)["error"]


def test_preview_endpoint_accepts_pad_s_and_returns_window(server):
    httpd, _url, _txt = server
    status, _h, body = _post(
        httpd, "/preview",
        {
            "toggles": [],
            "target": {"type": "line", "id": "0003"},
            "pad_s": 8,
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["pad_s"] == 8
    assert {"target_source_s", "center_s", "window_start_s", "window_end_s"} <= set(
        payload
    )


# ---- save（原子的書き込み・.bak・外部変更検出） ----

def test_save_creates_backup_once(tmp_path):
    session, txt = _session(tmp_path)
    backup = tmp_path / "input.sc.txt.bak"
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True
    session.save(toggles)
    assert backup.read_text(encoding="utf-8") == _TXT

    for t in toggles:
        if t["type"] == "line" and t["id"] == "0001":
            t["delete"] = True
    session.save(toggles)
    # .bak は初回の内容のまま（上書きしない）
    assert backup.read_text(encoding="utf-8") == _TXT


def test_save_twice_is_not_treated_as_external_change(tmp_path):
    """保存後に基準（raw TXT・mtime・ハッシュ）を更新する。"""
    session, txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True
    session.save(toggles)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0001":
            t["delete"] = True
    session.save(toggles)
    assert "[0001" not in txt.read_text(encoding="utf-8")


def test_save_detects_external_change(tmp_path):
    session, txt = _session(tmp_path)
    txt.write_text(_TXT + "[0005 0:09] 外部編集\n", encoding="utf-8")
    with pytest.raises(serve.ExternalChangeError):
        session.save(_all_toggles(tmp_path, session.raw_text))


def test_save_detects_external_change_with_preserved_mtime(tmp_path):
    """mtime を維持した内容変更（os.utime で偽装）でも検出される。"""
    session, txt = _session(tmp_path)
    stat_before = txt.stat()
    txt.write_text(_TXT + "[0005 0:09] 外部編集\n", encoding="utf-8")
    os.utime(txt, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    with pytest.raises(serve.ExternalChangeError):
        session.save(_all_toggles(tmp_path, session.raw_text))


def test_atomic_write_preserves_existing_permissions(tmp_path):
    path = tmp_path / "target.txt"
    path.write_text("before", encoding="utf-8")
    os.chmod(path, 0o600)
    serve._atomic_write(path, "after")
    assert path.read_text(encoding="utf-8") == "after"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_save_leaves_no_temp_file(tmp_path):
    session, _txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True
    session.save(toggles)
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".hlyst.tmp")]


def test_save_rerenders_review_wav(tmp_path):
    import soundfile as sf

    session, _txt = _session(tmp_path)
    before, _sr = sf.read(str(session.review_wav))
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True
    result = session.save(toggles)
    after, _sr = sf.read(str(session.review_wav))
    assert len(after) < len(before)
    assert result["version"] > 1
    assert any(line.startswith("Duration: ") for line in result["stats"])


def test_save_review_wav_equals_render_output(tmp_path):
    """保存後の review.wav は sc render の出力と一致する。"""
    import soundfile as sf

    session, txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True
    session.save(toggles)

    reference = render.render(str(txt), output=str(tmp_path / "ref.wav"))
    a, _sr = sf.read(str(session.review_wav))
    b, _sr2 = sf.read(reference)
    assert np.array_equal(a, b)


def test_save_audio_render_failure_still_saves_txt(tmp_path, monkeypatch):
    """review.wav の再生成に失敗しても TXT は保存済みのまま、専用エラーになる。"""
    session, txt = _session(tmp_path)
    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True

    def _boom(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(serve, "render_plan", _boom)
    with pytest.raises(serve.AudioRenderError) as exc:
        session.save(toggles)
    assert "保存済み" in str(exc.value)
    assert "[0003" not in txt.read_text(encoding="utf-8")


def test_save_error_does_not_touch_file(tmp_path):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, "[0001 0:00] あい\n")
    session = serve.HlystSession(str(txt))
    before = txt.read_bytes()
    with pytest.raises(serve.TxtEditError):
        session.save([{"type": "word", "id": "0001", "word": 1, "delete": False}])
    assert txt.read_bytes() == before


# ---- Range ヘッダの解釈 ----

@pytest.mark.parametrize("header,size,expected", [
    ("bytes=0-99", 1000, (0, 99)),
    ("bytes=100-", 1000, (100, 999)),
    ("bytes=-100", 1000, (900, 999)),
    ("bytes=999-999", 1000, (999, 999)),
    ("bytes=0-5000", 1000, (0, 999)),
    ("bytes=-5000", 1000, (0, 999)),
    (None, 1000, None),
    ("", 1000, None),
    ("items=0-9", 1000, None),
    ("bytes=0-9,20-29", 1000, None),
    ("bytes=1000-", 1000, "invalid"),
    ("bytes=500-100", 1000, "invalid"),
    ("bytes=abc", 1000, "invalid"),
    ("bytes=-0", 1000, "invalid"),
    ("bytes=-", 1000, "invalid"),
    ("bytes=0-99", 0, "invalid"),
])
def test_parse_byte_range(header, size, expected):
    assert serve.parse_byte_range(header, size) == expected


# ---- HTTP サーバー ----

@pytest.fixture
def server(tmp_path):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path)
    httpd, url = serve.create_server(str(txt))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, url, txt
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(httpd, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_address[1], timeout=10
    )
    try:
        conn.request(method, path, body=body, headers=headers or {})
        res = conn.getresponse()
        return res.status, dict(res.getheaders()), res.read()
    finally:
        conn.close()


def _token(httpd):
    return httpd.session.token


def _post(httpd, path, payload, token=None, origin=None, headers=None):
    token = _token(httpd) if token is None else token
    hdrs = {
        "Content-Type": "application/json",
        "X-Hlyst-Token": token,
        "Origin": origin or f"http://127.0.0.1:{httpd.server_address[1]}",
    }
    hdrs.update(headers or {})
    body = json.dumps(payload).encode("utf-8")
    return _request(httpd, "POST", f"{path}?token={token}", body, hdrs)


def test_server_binds_loopback_only(server):
    httpd, _url, _txt = server
    assert httpd.server_address[0] == "127.0.0.1"


def test_url_contains_token(server):
    httpd, url, _txt = server
    assert url.startswith(f"http://127.0.0.1:{httpd.server_address[1]}/?token=")
    assert _token(httpd) in url


def test_get_index_returns_html(server):
    httpd, _url, _txt = server
    status, headers, body = _request(httpd, "GET", f"/?token={_token(httpd)}")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Referrer-Policy"] == "no-referrer"
    text = body.decode("utf-8")
    assert "Hlyst" in text
    assert 'type="checkbox"' in text


def test_get_index_without_token_rejected(server):
    httpd, _url, _txt = server
    status, _h, _b = _request(httpd, "GET", "/")
    assert status == 403


def test_get_index_with_wrong_token_rejected(server):
    httpd, _url, _txt = server
    status, _h, _b = _request(httpd, "GET", "/?token=deadbeef")
    assert status == 403


def test_bad_host_header_rejected(server):
    httpd, _url, _txt = server
    status, _h, _b = _request(
        httpd, "GET", f"/?token={_token(httpd)}", headers={"Host": "evil.example"}
    )
    assert status == 403


def test_unknown_path_is_404(server):
    httpd, _url, _txt = server
    status, _h, _b = _request(httpd, "GET", f"/secrets?token={_token(httpd)}")
    assert status == 404


def test_path_traversal_is_404(server):
    httpd, _url, _txt = server
    status, _h, _b = _request(
        httpd, "GET", f"/audio/../../etc/passwd?token={_token(httpd)}"
    )
    assert status == 404


def test_audio_routes_are_allowlisted(server):
    httpd, _url, _txt = server
    for name in ("source", "review"):
        status, headers, body = _request(
            httpd, "GET", f"/audio/{name}?token={_token(httpd)}"
        )
        assert status == 200
        assert headers["Content-Type"] == "audio/wav"
        assert headers["Accept-Ranges"] == "bytes"
        assert body[:4] == b"RIFF"


def test_audio_range_request(server):
    httpd, _url, _txt = server
    status, headers, body = _request(
        httpd, "GET", f"/audio/review?token={_token(httpd)}",
        headers={"Range": "bytes=0-9"},
    )
    assert status == 206
    assert body == b"RIFF"[:4] + body[4:]
    assert len(body) == 10
    assert headers["Content-Range"].startswith("bytes 0-9/")


def test_audio_range_suffix_and_tail(server):
    httpd, _url, _txt = server
    _s, headers, full = _request(httpd, "GET", f"/audio/review?token={_token(httpd)}")
    size = len(full)
    status, headers, body = _request(
        httpd, "GET", f"/audio/review?token={_token(httpd)}",
        headers={"Range": "bytes=-16"},
    )
    assert status == 206
    assert body == full[-16:]
    assert headers["Content-Range"] == f"bytes {size - 16}-{size - 1}/{size}"

    status, headers, body = _request(
        httpd, "GET", f"/audio/review?token={_token(httpd)}",
        headers={"Range": f"bytes={size - 1}-"},
    )
    assert status == 206
    assert body == full[-1:]


def test_audio_range_invalid_returns_416(server):
    httpd, _url, _txt = server
    _s, _h, full = _request(httpd, "GET", f"/audio/review?token={_token(httpd)}")
    status, headers, body = _request(
        httpd, "GET", f"/audio/review?token={_token(httpd)}",
        headers={"Range": f"bytes={len(full)}-"},
    )
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{len(full)}"
    assert body == b""


def test_post_requires_token_header(server):
    httpd, _url, _txt = server
    status, _h, _b = _post(
        httpd, "/preview",
        {"toggles": [], "target": {"type": "line", "id": "0003"}},
        headers={"X-Hlyst-Token": "wrong"},
    )
    assert status == 403


def test_post_rejects_null_origin(server):
    httpd, _url, _txt = server
    status, _h, _b = _post(
        httpd, "/preview",
        {"toggles": [], "target": {"type": "line", "id": "0003"}},
        origin="null",
    )
    assert status == 403


def test_post_rejects_foreign_origin(server):
    httpd, _url, _txt = server
    status, _h, _b = _post(
        httpd, "/preview",
        {"toggles": [], "target": {"type": "line", "id": "0003"}},
        origin="http://evil.example",
    )
    assert status == 403


def test_post_requires_origin(server):
    httpd, _url, _txt = server
    token = _token(httpd)
    status, _h, _b = _request(
        httpd, "POST", f"/preview?token={token}", b"{}",
        {"Content-Type": "application/json", "X-Hlyst-Token": token},
    )
    assert status == 403


def test_post_rejects_oversized_body(server):
    httpd, _url, _txt = server
    token = _token(httpd)
    conn = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_address[1], timeout=10
    )
    try:
        conn.putrequest("POST", f"/preview?token={token}")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("X-Hlyst-Token", token)
        conn.putheader("Origin", f"http://127.0.0.1:{httpd.server_address[1]}")
        conn.putheader("Content-Length", str(serve.MAX_POST_BYTES + 1))
        conn.endheaders()
        res = conn.getresponse()
        assert res.status == 413
    finally:
        conn.close()


def test_post_rejects_unknown_segment_id(server):
    httpd, _url, _txt = server
    status, _h, body = _post(
        httpd, "/preview",
        {
            "toggles": [{"type": "line", "id": "9999", "delete": True}],
            "target": {"type": "line", "id": "0003"},
        },
    )
    assert status == 400
    assert "9999" in json.loads(body)["error"]


def test_post_unknown_path_is_404(server):
    httpd, _url, _txt = server
    status, _h, _b = _post(httpd, "/delete-everything", {"toggles": []})
    assert status == 404


def test_preview_endpoint_returns_audio(server):
    httpd, _url, txt = server
    before = txt.read_bytes()
    status, headers, body = _post(
        httpd, "/preview",
        {
            "toggles": [{"type": "word", "id": "0001", "word": 1, "delete": False}],
            "target": {"type": "word", "id": "0001", "word": 1},
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert base64.b64decode(payload["audio"])[:4] == b"RIFF"
    assert payload["center"] > 0
    word = _seg(payload["segments"], "0001")["words"][1]
    assert word["effective_delete"] is False
    assert txt.read_bytes() == before  # preview は保存しない


def test_save_endpoint_writes_txt_and_reloads(server):
    httpd, _url, txt = server
    session = httpd.session
    toggles = []
    for seg_id, value in serve.requested_state(
        session.raw_text, session.context.data, session.fillers
    )["segments"].items():
        toggles.append({
            "type": "line", "id": seg_id,
            "delete": True if seg_id == "0003" else value,
        })
    status, _h, body = _post(httpd, "/save", {"toggles": toggles})
    assert status == 200
    payload = json.loads(body)
    assert "[0003" not in txt.read_text(encoding="utf-8")
    assert payload["version"] >= 2
    assert _seg(payload["segments"], "0003")["effective_delete"] is True


def test_save_endpoint_reports_audio_render_failure(server, monkeypatch):
    """review.wav の再生成失敗は「保存失敗」ではなく txt_saved フラグ付きで返す。"""
    httpd, _url, txt = server
    session = httpd.session
    toggles = _all_toggles(_txt_dir(_txt_path(httpd)), session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0003":
            t["delete"] = True

    def _boom(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(serve, "render_plan", _boom)
    status, _h, body = _post(httpd, "/save", {"toggles": toggles})
    assert status == 500
    payload = json.loads(body)
    assert payload["txt_saved"] is True
    assert "保存済み" in payload["error"]
    assert "[0003" not in txt.read_text(encoding="utf-8")


def test_save_endpoint_reports_conflict(server):
    httpd, _url, txt = server
    txt.write_text("[0001 0:00] 外部編集\n", encoding="utf-8")
    status, _h, body = _post(httpd, "/save", {"toggles": []})
    assert status == 409
    assert "外部" in json.loads(body)["error"]


def test_save_endpoint_reports_manual_edit_error(server):
    httpd, _url, txt = server
    session = httpd.session
    session.raw_text = "[0001 0:00] あい\n"
    txt.write_text(session.raw_text, encoding="utf-8")
    session.digest = serve._sha256(session.raw_text)
    session.mtime_ns = txt.stat().st_mtime_ns
    status, _h, body = _post(
        httpd, "/save",
        {"toggles": [{"type": "word", "id": "0001", "word": 1, "delete": False}]},
    )
    assert status == 409
    assert "直接編集" in json.loads(body)["error"]


# ---- UTS 行のトグル ----

def test_uts_line_is_cut_by_default_and_kept_when_unchecked(tmp_path):
    """UTS 行は〔〕あり＝削除。チェックを外すと行を残して音声が復活する。"""
    session, txt = _session(tmp_path)
    data = session.review_data(session.raw_text, session.plan)
    seg = _seg(data["segments"], "0002")
    assert seg["toggle"] == "bracket"
    assert (seg["requested_delete"], seg["effective_delete"]) == (True, True)

    toggles = _all_toggles(tmp_path, session.raw_text)
    for t in toggles:
        if t["type"] == "line" and t["id"] == "0002":
            t["delete"] = False
    result = session.save(toggles)
    assert "[0002 0:02] 未転写発話 1.0秒\n" in txt.read_text(encoding="utf-8")
    seg = _seg(result["segments"], "0002")
    assert (seg["requested_delete"], seg["effective_delete"]) == (False, False)


def test_unsafe_uts_is_requested_but_kept(tmp_path):
    """カット境界が発話と連続する UTS は自動削除されず requested のみ True。"""
    _make_project(tmp_path)
    data = _data(tmp_path)
    # 島の前後の無音を消して「発話と連続」状態にする
    data["silences"] = [[1.0, 1.1], [1.6, 1.8], [4.8, 7.0]]
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    import soundfile as sf

    audio, sr = sf.read(str(tmp_path / "input.wav"))
    t = np.arange(int(round(1.0 * sr))) / sr
    audio[int(2.8 * sr): int(2.8 * sr) + len(t)] = 0.3 * np.sin(2 * np.pi * 550.0 * t)
    sf.write(str(tmp_path / "input.wav"), audio, sr, subtype="PCM_16")

    txt = _write_txt(tmp_path)
    session = serve.HlystSession(str(txt))
    seg = _seg(session.review_data(session.raw_text, session.plan)["segments"], "0002")
    assert (seg["requested_delete"], seg["effective_delete"]) == (True, False)


# ---- 画面スクリプトが期待する payload の形 ----

def test_payloads_contain_keys_used_by_client(server):
    """/preview /save のレスポンスに画面 JS が参照するキーが揃っている。"""
    httpd, _url, _txt = server
    session = httpd.session
    toggles = _all_toggles(_txt_dir(_txt_path(httpd)), session.raw_text)

    status, _h, body = _post(
        httpd, "/preview",
        {"toggles": toggles, "target": {"type": "word", "id": "0001", "word": 1}},
    )
    assert status == 200
    payload = json.loads(body)
    assert {"audio", "center", "message", "segments", "stats"} <= set(payload)
    word = _seg(payload["segments"], "0001")["words"][1]
    assert {"index", "requested_delete", "effective_delete", "output"} <= set(word)

    status, _h, body = _post(httpd, "/save", {"toggles": toggles})
    assert status == 200
    payload = json.loads(body)
    assert {"version", "txt_path", "segments", "stats", "warnings"} <= set(payload)
    assert isinstance(payload["stats"], list)


def test_review_audio_url_carries_token_query(server):
    """保存後のキャッシュバスター（&v=）が効くよう URL はクエリ付きで渡す。"""
    httpd, _url, _txt = server
    data = httpd.session.review_data(
        httpd.session.raw_text, httpd.session.plan
    )
    assert data["review_audio"].startswith("/audio/review?token=")
    assert data["source_audio"].startswith("/audio/source?token=")


def _txt_path(httpd):
    return httpd.session.context.txt_path


def _txt_dir(path):
    return path.parent


def test_preview_and_save_are_serialized_by_one_lock(tmp_path, monkeypatch):
    """preview / save（と保存基準の更新）は同じロックで直列化される。"""
    session, _txt = _session(tmp_path)
    seen = []
    original = serve.apply_toggles

    def _spy(*args, **kwargs):
        seen.append(session.lock.locked())
        return original(*args, **kwargs)

    monkeypatch.setattr(serve, "apply_toggles", _spy)
    toggles = _all_toggles(tmp_path, session.raw_text)
    session.preview(toggles, {"type": "line", "id": "0003"})
    session.save(toggles)
    assert seen == [True, True]
