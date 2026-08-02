import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from stefnceorf import render, review


# ---- chunk_output_starts（クロスフェード込みの累積） ----

def test_starts_single_chunk():
    starts, total = review.chunk_output_starts([(0, 100)], [], 10)
    assert starts == [0]
    assert total == 100


def test_starts_empty():
    starts, total = review.chunk_output_starts([], [], 10)
    assert starts == []
    assert total == 0


def test_starts_crossfade_shrinks_total():
    # 100 + 100 をフェード10で繋ぐ → 2つ目は 90 から始まり総長 190
    starts, total = review.chunk_output_starts([(0, 100), (200, 300)], [True], 10)
    assert starts == [0, 90]
    assert total == 190


def test_starts_simple_join_does_not_shrink():
    starts, total = review.chunk_output_starts([(0, 100), (200, 300)], [False], 10)
    assert starts == [0, 100]
    assert total == 200


def test_starts_fade_clamped_by_short_chunk():
    # 2つ目が5サンプルしかない → f = min(10, 100, 5) = 5
    starts, total = review.chunk_output_starts([(0, 100), (200, 205)], [True], 10)
    assert starts == [0, 95]
    assert total == 100


def test_starts_fade_clamped_by_accumulated_length():
    # 1つ目が3サンプル → f = min(10, 3, 100) = 3
    starts, total = review.chunk_output_starts([(0, 3), (200, 300)], [True], 10)
    assert starts == [0, 0]
    assert total == 100


def test_starts_mixed_flags_accumulate():
    bounds = [(0, 100), (200, 300), (400, 500)]
    starts, total = review.chunk_output_starts(bounds, [False, True], 10)
    assert starts == [0, 100, 190]
    assert total == 290


def test_starts_missing_flag_defaults_to_crossfade():
    # flags が足りない場合は crossfade_concat と同じくクロスフェード扱い
    starts, total = review.chunk_output_starts([(0, 100), (200, 300)], [], 10)
    assert starts == [0, 90]
    assert total == 190


# ---- segment_source_spans ----

def test_spans_uses_word_edges_and_unrecognized_source():
    segments = [
        {"id": "0001", "words": [
            {"word": "あ", "start": 0.0, "end": 1.0},
            {"word": "い", "start": 1.5, "end": 2.0},
        ]},
        {"id": "0002", "kind": "unrecognized", "source_start": 2.0,
         "source_end": 3.0, "words": []},
        {"id": "0003", "words": []},  # 音を持たない行は除外
    ]
    assert review.segment_source_spans(segments) == [
        ("0001", 0.0, 2.0), ("0002", 2.0, 3.0)
    ]


# ---- build_mapping（共通部分計算） ----

def test_mapping_intersection_within_chunk():
    spans = [("0001", 0.0, 1.0)]
    entries = review.build_mapping(spans, [(0, 16000)], [0], 16000)
    assert entries == [
        review.MapEntry("0001", 0, 16000, 0, 16000)
    ]


def test_mapping_clips_to_chunk_and_offsets_output():
    # セグメントは 0.5〜2.5s、チャンクは 1.0〜2.0s を出力 5000 から
    spans = [("0001", 0.5, 2.5)]
    entries = review.build_mapping(spans, [(16000, 32000)], [5000], 16000)
    assert entries == [
        review.MapEntry("0001", 16000, 32000, 5000, 21000)
    ]


def test_mapping_segment_split_into_two_entries():
    # 1セグメントが2チャンクへ分断される（間がカットされた）
    spans = [("0001", 0.0, 3.0)]
    bounds = [(0, 16000), (32000, 48000)]
    starts = [0, 16000]
    entries = review.build_mapping(spans, bounds, starts, 16000)
    assert [(e.output_start, e.output_end) for e in entries] == [
        (0, 16000), (16000, 32000)
    ]
    assert {e.segment_id for e in entries} == {"0001"}


def test_mapping_skips_non_overlapping_segments():
    spans = [("0001", 0.0, 1.0), ("0002", 5.0, 6.0)]
    entries = review.build_mapping(spans, [(0, 16000)], [0], 16000)
    assert [e.segment_id for e in entries] == ["0001"]


def test_mapping_follows_output_order_when_reordered():
    # 出力順が原音順と逆（並べ替え）でも、チャンク順にエントリが並ぶ
    spans = [("0001", 0.0, 1.0), ("0002", 1.0, 2.0)]
    bounds = [(16000, 32000), (0, 16000)]
    starts = [0, 16000]
    entries = review.build_mapping(spans, bounds, starts, 16000)
    assert [e.segment_id for e in entries] == ["0002", "0001"]
    assert entries[0].output_start == 0
    assert entries[1].output_start == 16000


# ---- map_source_sample ----

def test_map_inside_chunk():
    assert review.map_source_sample([(100, 200)], [0], 150) == 50


def test_map_before_first_chunk_returns_chunk_head():
    assert review.map_source_sample([(100, 200)], [0], 50) == 0


def test_map_after_last_chunk_returns_chunk_tail():
    assert review.map_source_sample([(100, 200)], [0], 300) == 100


def test_map_cut_region_returns_join_boundary():
    # 100-200 と 300-400 が残り、その間（カット済み）は結合境界へ写る
    bounds = [(100, 200), (300, 400)]
    starts = [0, 100]
    assert review.map_source_sample(bounds, starts, 210) == 100   # 前チャンク終端
    assert review.map_source_sample(bounds, starts, 290) == 100   # 後チャンク先頭
    assert review.map_source_sample(bounds, starts, 350) == 150


def test_map_empty_bounds_returns_none():
    assert review.map_source_sample([], [], 10) is None


# ---- _embed_json ----

def test_embed_json_escapes_script_close():
    html = review._embed_json({"t": "危険</script><script>alert(1)"})
    assert "</script>" not in html
    assert json.loads(html)["t"] == "危険</script><script>alert(1)"


def test_embed_json_escapes_html_comment_open():
    html = review._embed_json({"t": "<!-- x -->"})
    assert "<!--" not in html
    assert json.loads(html)["t"] == "<!-- x -->"


def test_embed_json_keeps_japanese_raw():
    assert "あ" in review._embed_json({"t": "あ"})


# ---- 実プロジェクト（合成wav + 手書き json） ----

def _make_project(tmp_path, sr=16000):
    """レビュー検証用の最小プロジェクト。

    0001 speech: あ[0,1)220 / えー[1.1,1.6)440(suggest) / い[1.8,2.65)660
    （2.65〜2.8 は既知無音。0002 削除時のガード用ポーズに使う）
    0002 unrecognized(uts): [2.8,3.8)
    （3.8〜3.95 も既知無音。ガード用ポーズの残り半分に使う）
    0003 speech: う[3.95,4.8)880
    （長無音 4.8〜7.0 = 2.2s > gap_threshold → 無音トリム対象）
    0004 speech: え[7.0,8.0)330
    """
    import soundfile as sf

    total_s = 8.0
    audio = np.zeros(int(round(total_s * sr)))

    def _tone(start, end, freq):
        t = np.arange(int(round((end - start) * sr))) / sr
        a = int(round(start * sr))
        audio[a: a + len(t)] = 0.3 * np.sin(2 * np.pi * freq * t)

    _tone(0.0, 1.0, 220.0)
    _tone(1.1, 1.6, 440.0)
    _tone(1.8, 2.65, 660.0)
    _tone(2.8, 3.8, 550.0)
    _tone(3.95, 4.8, 880.0)
    _tone(7.0, 8.0, 330.0)

    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "pause_threshold": 0.15,
        "silences": [
            [1.0, 1.1], [1.6, 1.8], [2.65, 2.8], [3.8, 3.95], [4.8, 7.0],
        ],
        # guarded cut（未認識/UTS）は細かいRMSパスの無音だけを使う
        "rms_silences": [
            [1.0, 1.1], [1.6, 1.8], [2.65, 2.8], [3.8, 3.95], [4.8, 7.0],
        ],
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
                    {"word": "い", "start": 1.8, "end": 2.65,
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
                    {"word": "う", "start": 3.95, "end": 4.8,
                     "probability": 0.9, "block": 3},
                ],
            },
            {
                "id": "0004",
                "text": "え",
                "words": [
                    {"word": "え", "start": 7.0, "end": 8.0,
                     "probability": 0.9, "block": 4},
                ],
            },
        ],
    }
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav


_FULL_TXT = (
    "[0001] あえーい\n"
    "[0002] ⚠ 未転写発話候補 1.0秒（音声保持）\n"
    "[0003] う\n"
    "[0004] え\n"
)


def _write_txt(tmp_path, body):
    txt = tmp_path / "input.sc.txt"
    txt.write_text(body, encoding="utf-8")
    return txt


# ---- 写像表の総出力長 == 実レンダー出力長（サンプル単位） ----

@pytest.mark.parametrize("body", [
    _FULL_TXT,
    # 行削除（未認識行）
    "[0001] あえーい\n[0003] う\n[0004] え\n",
    # フィラー〔〕削除
    "[0001] あ〔えー〕い\n[0002] ⚠\n[0003] う\n[0004] え\n",
    # 単語の手動削除
    "[0001] あえー\n[0002] ⚠\n[0003] う\n[0004] え\n",
    # 並べ替え
    "[0003] う\n[0001] あえーい\n[0002] ⚠\n[0004] え\n",
    # 複数行削除
    "[0001] あえーい\n[0004] え\n",
    # 1行だけ
    "[0003] う\n",
])
def test_mapping_total_matches_real_render_samples(tmp_path, body):
    """写像表から算出した総出力サンプル数が実レンダー出力と一致する。"""
    import soundfile as sf

    _make_project(tmp_path)
    txt = _write_txt(tmp_path, body)

    plan = render.build_edit_plan(str(txt))
    out = render.render_plan(plan, tmp_path / "probe.wav")
    audio, _sr = sf.read(out)

    assert review.mapping_total_samples(plan) == len(audio)


def test_review_data_output_samples_matches_review_wav(tmp_path):
    """review() が書いた review.wav の長さと埋め込みデータの output_samples が一致。"""
    import soundfile as sf

    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)
    plan = render.build_edit_plan(str(txt))
    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))

    wav = render.render_plan(plan, tmp_path / "input.review.wav")
    audio, _sr = sf.read(wav)
    rd = review.build_review_data(
        plan, data, {"0001", "0002", "0003", "0004"}, "input.wav",
        "input.review.wav",
    )
    assert rd["output_samples"] == len(audio)


# ---- build_review_data ----

def _build(tmp_path, body):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, body)
    plan = render.build_edit_plan(str(txt))
    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))
    kept = {sid for sid, _t, _b in render.parse_edited_txt(
        txt.read_text(encoding="utf-8"))}
    return plan, review.build_review_data(
        plan, data, kept, "input.wav", "input.review.wav"
    )


def _seg(rd, seg_id):
    return next(s for s in rd["segments"] if s["id"] == seg_id)


def test_review_data_lists_all_json_segments(tmp_path):
    _plan, rd = _build(tmp_path, "[0001] あえーい\n[0003] う\n")
    assert [s["id"] for s in rd["segments"]] == ["0001", "0002", "0003", "0004"]


def test_review_data_marks_deleted_segments(tmp_path):
    _plan, rd = _build(tmp_path, "[0001] あえーい\n[0003] う\n")
    assert _seg(rd, "0002")["deleted"] is True
    assert _seg(rd, "0004")["deleted"] is True
    assert _seg(rd, "0001")["deleted"] is False


def test_review_data_marks_uts_and_label(tmp_path):
    _plan, rd = _build(tmp_path, _FULL_TXT)
    seg = _seg(rd, "0002")
    assert seg["kind"] == "unrecognized"
    assert seg["reason"] == "uts"
    assert "〔未転写発話" in seg["label"]


def test_review_data_keeps_suggest_flag(tmp_path):
    _plan, rd = _build(tmp_path, _FULL_TXT)
    words = _seg(rd, "0001")["words"]
    assert [w["suggest"] for w in words] == [False, True, False]


def test_review_data_no_edit_keeps_all_words(tmp_path):
    _plan, rd = _build(tmp_path, _FULL_TXT)
    assert all(not w["cut"] for w in _seg(rd, "0001")["words"])


def test_review_data_marks_cut_filler_word(tmp_path):
    """〔〕でフィラーを削除すると、その単語だけ cut になる。"""
    _plan, rd = _build(
        tmp_path, "[0001] あ〔えー〕い\n[0002] ⚠\n[0003] う\n[0004] え\n"
    )
    words = _seg(rd, "0001")["words"]
    assert [w["cut"] for w in words] == [False, True, False]


def test_review_data_output_time_is_earlier_after_cut(tmp_path):
    """フィラーカット後、後続単語の編集後時刻は原音時刻より前へずれる。"""
    _plan, rd = _build(
        tmp_path, "[0001] あ〔えー〕い\n[0002] ⚠\n[0003] う\n[0004] え\n"
    )
    last = _seg(rd, "0001")["words"][-1]
    assert last["output"] < last["source"]


def test_review_data_first_segment_maps_to_zero(tmp_path):
    _plan, rd = _build(tmp_path, _FULL_TXT)
    assert _seg(rd, "0001")["output"] == pytest.approx(0.0, abs=1e-9)


def test_review_data_deleted_segment_uses_join_boundary(tmp_path):
    """削除行の編集後時刻は、後続の生存行の出力開始（＝結合境界）になる。"""
    _plan, rd = _build(tmp_path, "[0001] あえーい\n[0003] う\n[0004] え\n")
    assert _seg(rd, "0002")["output"] == pytest.approx(
        _seg(rd, "0003")["output"], abs=1e-9
    )


def test_review_data_trailing_deleted_segment_uses_previous_end(tmp_path):
    """末尾の削除行は後続が無いので、直前の生存行の出力終端を境界にする。

    0003（う[3.8,4.8]）が最後の生存行。その原音区間の終端 4.8 が編集後でも 4.8
    （0003 より前はカットされていない）で、出力全長（末尾マージン込み 5.0）内に収まる。
    """
    _plan, rd = _build(tmp_path, "[0001] あえーい\n[0002] ⚠\n[0003] う\n")
    seg = _seg(rd, "0004")
    assert seg["deleted"] is True
    assert seg["output"] == pytest.approx(4.8, abs=0.02)
    assert _seg(rd, "0003")["output"] < seg["output"] <= rd["output_duration"]


def test_review_data_reorder_reflected_in_output_times(tmp_path):
    """並べ替えると編集後時刻も出力順に従う。"""
    _plan, rd = _build(
        tmp_path, "[0003] う\n[0001] あえーい\n[0002] ⚠\n[0004] え\n"
    )
    assert _seg(rd, "0003")["output"] < _seg(rd, "0001")["output"]
    assert _seg(rd, "0003")["source"] > _seg(rd, "0001")["source"]


def test_review_data_reports_silence_trim(tmp_path):
    """4.8〜7.0 の長無音が切り詰められ、trims に記録される。

    既知無音は前後区間のマージン込み境界 [5.0, 6.98] にクリップされ、その両端に
    gap_max/2=0.35 ずつ残すので、カット範囲は 5.35〜6.63。
    """
    _plan, rd = _build(tmp_path, _FULL_TXT)
    assert len(rd["trims"]) == 1
    trim = rd["trims"][0]
    assert trim["source"] == pytest.approx(5.35, abs=0.02)
    assert trim["source_end"] == pytest.approx(6.63, abs=0.02)
    assert trim["removed"] == pytest.approx(1.28, abs=0.05)


def test_review_data_carries_stats_and_note(tmp_path):
    _plan, rd = _build(tmp_path, _FULL_TXT)
    assert any(line.startswith("Duration: ") for line in rd["stats"])
    assert "sc render" in rd["note"]
    assert "Logic" in rd["note"]


# ---- HTML 生成 ----

def _html(tmp_path, body=_FULL_TXT):
    _plan, rd = _build(tmp_path, body)
    return review.render_html(rd), rd


def _transcript_html(html):
    """#transcript のマークアップ部分だけを取り出す（埋め込み JSON を除く）。"""
    body = html.split('<div id="transcript">')[1]
    return body.split('\n</div>\n<script id="sc-data"')[0]


def _payload_json(html):
    """埋め込み JSON をパースして返す。"""
    payload = html.split('<script id="sc-data" type="application/json">')[1]
    return json.loads(payload.split("</script>")[0])


def test_html_is_self_contained_without_cdn(tmp_path):
    html, _rd = _html(tmp_path)
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src=" not in html
    assert "<link " not in html


def test_html_embeds_review_data(tmp_path):
    html, rd = _html(tmp_path)
    assert '<script id="sc-data" type="application/json">' in html
    parsed = _payload_json(html)
    assert [s["id"] for s in parsed["segments"]] == [
        "0001", "0002", "0003", "0004"
    ]
    assert parsed["output_samples"] == rd["output_samples"]


def test_html_escapes_script_close_in_transcript_text(tmp_path):
    """単語テキストに </script> が含まれても JSON/マークアップを壊さない。"""
    _make_project(tmp_path)
    json_path = tmp_path / "input.sc.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["segments"][0]["words"][0]["word"] = "</script><img src=x>"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    txt = _write_txt(tmp_path, _FULL_TXT)
    plan = render.build_edit_plan(str(txt))
    rd = review.build_review_data(
        plan, data, {"0001", "0002", "0003", "0004"},
        "input.wav", "input.review.wav",
    )
    html = review.render_html(rd)

    # 埋め込み JSON は </script> で途切れず、値は元のまま復元できる
    assert _payload_json(html)["segments"][0]["words"][0]["text"] == (
        "</script><img src=x>"
    )
    # マークアップ側は HTML エスケープされ、タグとして解釈されない
    transcript = _transcript_html(html)
    assert "<img src=x>" not in transcript
    assert "&lt;/script&gt;&lt;img src=x&gt;" in transcript


def test_html_marks_deleted_line_with_strikethrough(tmp_path):
    html, _rd = _html(tmp_path, "[0001] あえーい\n[0003] う\n")
    assert 'class="line deleted' in _transcript_html(html)
    assert ".line.deleted .text { text-decoration: line-through" in html


def test_html_kept_line_not_marked_deleted(tmp_path):
    html, _rd = _html(tmp_path, _FULL_TXT)
    assert "deleted" not in _transcript_html(html)


def test_html_marks_suggest_and_cut_words(tmp_path):
    html, _rd = _html(
        tmp_path, "[0001] あ〔えー〕い\n[0002] ⚠\n[0003] う\n[0004] え\n"
    )
    assert 'class="w suggest cut"' in _transcript_html(html)


def test_html_uses_relative_audio_paths(tmp_path):
    html, _rd = _html(tmp_path)
    assert 'id="audio-src" controls preload="metadata" src="input.wav"' in html
    assert 'id="audio-out" controls preload="metadata" src="input.review.wav"' in html
    assert str(tmp_path) not in html


def test_html_shows_logic_note(tmp_path):
    html, _rd = _html(tmp_path)
    assert "Logic 上の再生とは異なります" in html


def test_html_shows_silence_trim_marker(tmp_path):
    html, _rd = _html(tmp_path)
    assert 'class="trim" data-t="0"' in html
    assert "無音トリム" in html


def test_html_lists_every_segment_line(tmp_path):
    html, _rd = _html(tmp_path)
    for i in range(4):
        assert f'data-i="{i}"' in html


def test_html_word_indices_present(tmp_path):
    html, _rd = _html(tmp_path)
    assert 'data-w="0"' in html
    assert 'data-w="2"' in html


# ---- _relative_media ----

def test_relative_media_same_dir(tmp_path):
    assert review._relative_media(tmp_path / "a.wav", tmp_path) == "a.wav"


def test_relative_media_parent_dir(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    assert review._relative_media(tmp_path / "a.wav", sub) == "../a.wav"


def test_relative_media_percent_encodes_spaces(tmp_path):
    assert review._relative_media(tmp_path / "a b.wav", tmp_path) == "a%20b.wav"


# ---- review() 統合 ----

def test_review_creates_wav_and_html(tmp_path):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)

    res = review.review(str(txt), open_browser=False)

    assert res["wav_path"] == str(tmp_path / "input.review.wav")
    assert res["html_path"] == str(tmp_path / "input.review.html")
    assert (tmp_path / "input.review.wav").exists()
    assert (tmp_path / "input.review.html").exists()


def test_review_does_not_touch_edited_wav(tmp_path):
    """既存の .edited.wav / .logic.fcpxml を上書きしない。"""
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)
    edited = tmp_path / "input.edited.wav"
    edited.write_bytes(b"sentinel")
    fcpxml = tmp_path / "input.logic.fcpxml"
    fcpxml.write_bytes(b"sentinel-xml")

    review.review(str(txt), open_browser=False)

    assert edited.read_bytes() == b"sentinel"
    assert fcpxml.read_bytes() == b"sentinel-xml"


def test_review_wav_equals_render_output(tmp_path):
    """レビュー音声は sc render と同じ内容（同じ EditPlan からの実レンダー）。"""
    import soundfile as sf

    _make_project(tmp_path)
    txt = _write_txt(tmp_path, "[0001] あえーい\n[0003] う\n")

    review.review(str(txt), open_browser=False)
    rendered = render.render(str(txt), output=str(tmp_path / "ref.wav"))

    a, sr_a = sf.read(str(tmp_path / "input.review.wav"))
    b, sr_b = sf.read(rendered)
    assert sr_a == sr_b
    assert np.array_equal(a, b)


def test_review_opens_browser_by_default(tmp_path, monkeypatch):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)
    opened = []
    monkeypatch.setattr(review.webbrowser, "open", lambda url: opened.append(url))

    review.review(str(txt))

    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith("input.review.html")


def test_review_no_open_skips_browser(tmp_path, monkeypatch):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)

    def _fail(url):
        raise AssertionError("ブラウザを開いてはいけない")

    monkeypatch.setattr(review.webbrowser, "open", _fail)
    review.review(str(txt), open_browser=False)


def test_review_stats_out_suppresses_print(tmp_path, capsys):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)
    stats = []
    review.review(str(txt), open_browser=False, stats_out=stats)
    assert any(line.startswith("Duration: ") for line in stats)
    assert capsys.readouterr().out == ""


def test_review_prints_stats_without_stats_out(tmp_path, capsys):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)
    review.review(str(txt), open_browser=False)
    assert "Duration: " in capsys.readouterr().out


def test_review_missing_txt_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        review.review(str(tmp_path / "nope.sc.txt"), open_browser=False)


def test_review_unknown_id_raises(tmp_path):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, "[9999] なし\n")
    with pytest.raises(ValueError):
        review.review(str(txt), open_browser=False)


# ---- 状態モデル（requested / effective）と serve モードの HTML ----

def test_state_label_combinations():
    assert review.state_label(True, True) == "削除"
    assert review.state_label(True, False) == "unsafeのため保持"
    assert review.state_label(False, False) == "保持"
    assert review.state_label(False, True) == "削除（手動編集）"


def test_state_label_unrecognized_effective_delete_is_just_deleted():
    """未認識/UTS も guarded cut で削除できた場合は他の行と同じ「削除」にする。

    未認識/UTS の削除は元位置ポーズ0.25秒を条件とする guarded cut になった
    ため、無印の未認識行専用の「削除（無条件）」文言は廃止した。
    """
    assert review.state_label(True, True, kind="unrecognized", reason=None) == "削除"
    assert review.state_label(True, True, kind="unrecognized", reason="uts") == "削除"
    assert review.state_label(True, True, kind="speech", reason=None) == "削除"


def test_plain_unrecognized_uses_safety_label_when_pause_is_short():
    assert review.state_label(
        True, False, kind="unrecognized", reason=None,
        unsafe_reason="pause",
    ) == "ポーズ不足のため保持"


def test_review_data_reports_requested_and_effective(tmp_path):
    """既定（requested 未指定）では TXT の行の有無と写像の被覆から推定する。"""
    _plan, rd = _build(tmp_path, "[0001] あえーい\n[0003] う\n")
    assert _seg(rd, "0002")["requested_delete"] is True
    assert _seg(rd, "0002")["effective_delete"] is True
    assert _seg(rd, "0001")["requested_delete"] is False
    assert _seg(rd, "0001")["effective_delete"] is False


def test_review_data_requested_override_is_used(tmp_path):
    """requested を渡すと（unsafe 等で）effective と食い違っても保持される。"""
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, _FULL_TXT)
    plan = render.build_edit_plan(str(txt))
    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))
    rd = review.build_review_data(
        plan, data, {"0001", "0002", "0003", "0004"},
        "input.wav", "input.review.wav",
        serve=True,
        requested={"segments": {"0001": True}, "words": {"0001": {1: True}}},
    )
    assert _seg(rd, "0001")["requested_delete"] is True
    assert _seg(rd, "0001")["effective_delete"] is False
    assert _seg(rd, "0001")["words"][1]["requested_delete"] is True
    assert _seg(rd, "0001")["words"][1]["effective_delete"] is False


def test_review_data_uts_toggle_kind(tmp_path):
    _plan, rd = _build(tmp_path, _FULL_TXT)
    assert _seg(rd, "0002")["toggle"] == "bracket"
    assert _seg(rd, "0001")["toggle"] == "line"


def test_html_title_is_hlyst(tmp_path):
    html, _rd = _html(tmp_path)
    assert "<title>Hlyst — input</title>" in html
    assert "A/B試聴" in html


def test_static_html_has_no_checkboxes(tmp_path):
    html, _rd = _html(tmp_path)
    assert 'type="checkbox"' not in html
    assert 'id="save"' not in html


def _serve_html(tmp_path, body=_FULL_TXT, requested=None):
    _make_project(tmp_path)
    txt = _write_txt(tmp_path, body)
    plan = render.build_edit_plan(str(txt))
    data = json.loads((tmp_path / "input.sc.json").read_text(encoding="utf-8"))
    kept = {sid for sid, _t, _b in render.parse_edited_txt(
        txt.read_text(encoding="utf-8"))}
    rd = review.build_review_data(
        plan, data, kept, "/audio/source?token=t", "/audio/review?token=t",
        serve=True, requested=requested, token="t",
    )
    return review.render_html(rd), rd


def test_serve_html_has_line_and_word_checkboxes(tmp_path):
    html, _rd = _serve_html(tmp_path)
    transcript = _transcript_html(html)
    assert '<input type="checkbox" class="del" data-i="0"' in transcript
    assert 'class="delw" data-i="0" data-w="1"' in transcript
    # フィラー提案でない単語にはチェックボックスを付けない
    assert 'data-w="0" aria-label' not in transcript


def test_serve_html_checkbox_reflects_requested_state(tmp_path):
    html, _rd = _serve_html(
        tmp_path, "[0001] あ〔えー〕い\n[0002] ⚠\n[0003] う\n[0004] え\n"
    )
    transcript = _transcript_html(html)
    assert 'class="delw" data-i="0" data-w="1" aria-label="このフィラーを削除" checked' \
        in transcript


def test_serve_html_shows_state_labels(tmp_path):
    html, _rd = _serve_html(
        tmp_path, "[0001] あ〔えー〕い\n[0002] ⚠\n[0003] う\n",
    )
    transcript = _transcript_html(html)
    assert '<span class="state" data-i="0">保持</span>' in transcript
    assert '<span class="state" data-i="3">削除</span>' in transcript


def test_serve_html_shows_unsafe_state(tmp_path):
    html, _rd = _serve_html(
        tmp_path, _FULL_TXT, requested={"segments": {"0001": True}, "words": {}}
    )
    assert "unsafeのため保持" in _transcript_html(html)


def test_serve_html_includes_save_controls_and_script(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert 'id="save"' in html
    assert 'id="audio-preview"' in html
    assert '"/preview"' in html
    assert '"/save"' in html
    assert "X-Hlyst-Token" in html


def test_serve_html_has_no_cdn(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src=" not in html


def test_serve_html_embeds_token(tmp_path):
    html, rd = _serve_html(tmp_path)
    assert rd["token"] == "t"
    assert _payload_json(html)["serve"] is True
    assert _payload_json(html)["token"] == "t"


# ---- UI 改善（Flexoki 配色 / unsafe 表現 / トグル時の再生方式 / シーク / 2行レイアウト） ----

def test_html_uses_flexoki_custom_properties(tmp_path):
    html, _rd = _html(tmp_path)
    assert "--bg: #FFFCF0" in html
    assert "--red: #AF3029" in html
    assert "prefers-color-scheme: dark" in html
    assert "--accent: #4385BE" in html
    # 直書きの rgba/hex はカスタムプロパティ参照に置き換わっている
    assert "rgba(10,100,200" not in html
    assert "rgba(220,60,40" not in html
    assert "#0a64c8" not in html


def test_unsafe_is_outline_not_background_fill(tmp_path):
    """削除指定→安全判定で保持は塗りつぶしではなく破線アウトライン＋赤太字ラベル。"""
    html, _rd = _html(tmp_path)
    assert ".w.unsafe, .line.unsafe {" in html
    assert "outline: 1px dashed var(--red)" in html
    assert ".state.unsafe { color: var(--red); font-weight: 700; }" in html


def test_serve_html_state_span_has_unsafe_class(tmp_path):
    html, _rd = _serve_html(
        tmp_path, _FULL_TXT, requested={"segments": {"0001": True}, "words": {}}
    )
    transcript = _transcript_html(html)
    assert 'class="state unsafe" data-i="0"' in transcript


def test_segment_markup_shows_deleted_label_for_plain_unrecognized():
    """UTS でない無印の未認識行も、実際に削除できれば「削除」と表示する。"""
    seg = {
        "index": 0, "id": "0002", "kind": "unrecognized", "reason": None,
        "effective_delete": True, "requested_delete": True, "deleted": True,
        "ts": "00:02", "words": [], "label": "⚠ 未認識区間 1.0秒（音声保持）",
    }
    markup = review._segment_markup(seg, serve=True)
    assert '<span class="state" data-i="0">削除</span>' in markup


def test_segment_markup_shows_pause_label_when_unsafe_reason_is_pause():
    """unsafe_reason が pause のときは「ポーズ不足のため保持」を表示する。"""
    seg = {
        "index": 0, "id": "0002", "kind": "unrecognized", "reason": None,
        "effective_delete": False, "requested_delete": True, "deleted": True,
        "unsafe_reason": "pause",
        "ts": "00:02", "words": [], "label": "⚠ 未認識区間 1.0秒（音声保持）",
    }
    markup = review._segment_markup(seg, serve=True)
    assert '<span class="state unsafe" data-i="0">ポーズ不足のため保持</span>' in markup


def test_serve_script_plays_preview_immediately_on_response(tmp_path):
    """応答受信時に audio-preview へ src 設定・load・play を試行する。

    autoplay 拒否は握りつぶし、src 設定と load() までは完了させる
    （再生ボタンを押せばすぐ聞ける状態にするため）。
    """
    html, _rd = _serve_html(tmp_path)
    run_preview_body = html.split("function runPreview(target) {")[1].split(
        "\n  }\n"
    )[0]
    assert "previewCache[key] = entry;" in run_preview_body
    assert "preview.src = url;" in run_preview_body
    assert "preview.load();" in run_preview_body
    assert "var playing = preview.play();" in run_preview_body
    assert "playing.catch(function () {});" in run_preview_body
    # クリック時に再生するコードパスも維持している（別ハンドラでキャッシュを再生する）
    assert "var cached = previewCache[key];" in html


def test_serve_script_disables_checkboxes_while_busy(tmp_path):
    """busy 中はチェックボックス（.del/.delw）と保存ボタンを操作不能にする。"""
    html, _rd = _serve_html(tmp_path)
    assert "input.del, input.delw" in html
    assert "function setControlsBusy(value) {" in html
    assert "box.disabled = value;" in html


def test_serve_script_guards_stale_responses_with_generation(tmp_path):
    """世代IDが最新でない応答は mergeStates/applyStates/キャッシュ格納を行わない。"""
    html, _rd = _serve_html(tmp_path)
    assert "var generation = 0;" in html
    assert "var myGeneration = ++generation;" in html
    assert "if (myGeneration !== generation) { return; }" in html


def test_serve_script_clears_preview_cache_on_toggle_and_save(tmp_path):
    """任意のトグル変更時と保存成功時に previewCache を全破棄する。"""
    html, _rd = _serve_html(tmp_path)
    assert "function clearPreviewCache() {" in html
    assert "URL.revokeObjectURL(previewCache[key].url);" in html
    # change ハンドラ（トグル）と保存成功時の両方で呼ばれている
    assert html.count("clearPreviewCache();") >= 3


def test_html_has_seek_buttons_for_both_players(tmp_path):
    html, _rd = _html(tmp_path)
    assert html.count('data-player="src"') == 4
    assert html.count('data-player="out"') == 4
    assert 'data-seek="-10"' in html
    assert 'data-seek="-5"' in html
    assert 'data-seek="5"' in html
    assert 'data-seek="10"' in html


def test_html_controls_is_two_player_rows(tmp_path):
    html, _rd = _html(tmp_path)
    assert html.count('class="player-row"') == 2


# ---- 試聴範囲切替・絶対時刻表示・保存ボタン移設・クリック挙動（UI改修） ----

def test_serve_html_has_pad_range_buttons_default_4s(tmp_path):
    html, _rd = _serve_html(tmp_path)
    transcript_area = html.split('<div class="preview-panel">')[1].split(
        "</div>"
    )[0]
    assert 'data-pad="2"' in transcript_area
    assert 'data-pad="4"' in transcript_area
    assert 'data-pad="8"' in transcript_area
    assert 'class="padbtn active" data-pad="4"' in transcript_area


def test_serve_html_has_absolute_time_placeholders(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert 'id="abs-target"' in html
    assert 'id="abs-center"' in html
    assert 'id="abs-window"' in html
    assert 'id="abs-pos"' in html


def test_serve_html_save_button_is_in_first_player_row(tmp_path):
    """保存ボタンは3段目ではなく1段目（原音プレイヤー行）にある。"""
    html, _rd = _serve_html(tmp_path)
    row1 = html.split('class="player-row"')[1].split('class="player-row"')[0]
    assert 'id="save"' in row1
    preview_panel = html.split('<div class="preview-panel">')[1].split("</div>")[0]
    assert 'id="save"' not in preview_panel


def test_serve_html_save_button_default_label(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert '<button type="button" id="save" disabled>保存済み</button>' in html


def test_serve_script_save_button_shows_dirty_count(tmp_path):
    """保存ボタンの表示は dirty 件数（N件）で更新される。"""
    html, _rd = _serve_html(tmp_path)
    assert '"変更を保存（" + n + "件）"' in html
    assert '"保存済み"' in html
    assert '"保存中…"' in html


def test_serve_script_normal_click_uses_audio_preview(tmp_path):
    """通常クリックは3段目「現在案プレビュー」（audio-preview）を再生する。"""
    html, _rd = _serve_html(tmp_path)
    assert "function playCurrentPlan(target) {" in html
    assert "preview.src = cached.url;" in html
    serve_script = html.split("if (!DATA.serve) { return; }")[1]
    assert "playCurrentPlan(target);" in serve_script


def test_serve_script_has_shift_and_alt_shortcuts(tmp_path):
    """Shift+クリック=保存済み編集後、Option(Alt)+クリック=原音のショートカットがある。"""
    html, _rd = _serve_html(tmp_path)
    assert "ev.shiftKey" in html
    assert "ev.altKey" in html
    assert "seekAndPlay(audioOut," in html
    assert "seekAndPlay(audioSrc," in html
    # Ctrl/Cmd は使わない
    assert "ev.ctrlKey" not in html
    assert "ev.metaKey" not in html


def test_serve_script_playback_position_uses_window_start_plus_current_time(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert (
        "absPos.textContent = formatTime(currentWindowStart + preview.currentTime);"
        in html
    )


def test_serve_script_saved_player_not_updated_by_preview(tmp_path):
    """保存済み編集後プレイヤー（audio-out）の src は保存成功時にしか更新されない。"""
    html, _rd = _serve_html(tmp_path)
    run_preview_body = html.split("function runPreview(target) {")[1].split(
        "\n  }\n"
    )[0]
    assert "audio-out" not in run_preview_body
    assert html.count('out.src = DATA.review_audio + "&v="') == 1


def test_serve_html_extra_legend_mentions_shortcuts(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert "Shift" in html
    assert "Option" in html


# ---- 保存済み・現在案・削除指定の状態分離（Hlyst UX 改修 Task 1） ----

def test_preview_merge_does_not_overwrite_requested_or_saved_positions(tmp_path):
    html, _rd = _serve_html(tmp_path)
    body = html.split("function mergePreviewStates(segments) {")[1].split("\n  }")[0]
    assert "effective_delete" in body
    assert "requested_delete" not in body
    assert "savedPositions" not in body


def test_review_data_exposes_center_positions(tmp_path):
    _html, rd = _serve_html(tmp_path)
    seg = rd["segments"][0]
    assert {"source_center", "output_center"} <= set(seg)
    if seg["words"]:
        assert {"center", "output_center"} <= set(seg["words"][0])


# ---- preview範囲変更の競合をなくす（Hlyst UX 改修 Task 2） ----

def test_preview_freezes_pad_for_request_and_cache_key(tmp_path):
    html, _rd = _serve_html(tmp_path)
    body = html.split("function runPreview(target) {")[1].split("\n  }")[0]
    assert "var requestPad = currentPad;" in body
    assert "pad_s: requestPad" in body
    assert "previewKey(target, requestPad)" in body


def test_busy_disables_pad_buttons(tmp_path):
    html, _rd = _serve_html(tmp_path)
    body = html.split("function setControlsBusy(value")[1].split("\n  }")[0]
    assert ".padbtn" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node が見つからない")
def test_generated_scripts_are_valid_javascript(tmp_path):
    """生成 HTML 内の <script> はどちらも node --check で構文検証できる。"""
    html, _rd = _serve_html(tmp_path)
    checked = 0
    for match in re.finditer(r'<script(?:\s[^>]*)?>(.*?)</script>', html, re.S):
        attrs_body = match.group(0)
        if 'type="application/json"' in attrs_body:
            continue
        code = match.group(1)
        js_path = tmp_path / f"script_{checked}.js"
        js_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            ["node", "--check", str(js_path)], capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        checked += 1
    assert checked == 2
