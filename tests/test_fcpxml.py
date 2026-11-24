"""Logic Pro 向け FCPXML 出力のテスト。"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from stefnceorf import render
from stefnceorf.fcpxml import (
    EDGE_SAFE_DBFS,
    EDGE_SEARCH_S,
    export_fcpxml,
    _resolve_forward_overlaps,
    _snap_edge_inward,
    _snap_region,
    _timeline_offsets,
)

_SAFE_AMP = 10 ** (EDGE_SAFE_DBFS / 20.0)


def _write_project(
    root: Path,
    *,
    channels: int = 1,
    samplerate: int = 44100,
    segments: list[dict] | None = None,
    txt: str = "[0001] 音声\n",
    extra: dict | None = None,
) -> tuple[Path, Path]:
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    wav = root / "元 音声.wav"
    frames = samplerate * 6
    audio = np.zeros(frames) if channels == 1 else np.zeros((frames, channels))
    sf.write(wav, audio, samplerate, subtype="PCM_16")
    if segments is None:
        segments = [
            {
                "id": "0001",
                "text": "音声",
                "words": [
                    {
                        "word": "音声",
                        "start": 0.123456,
                        "end": 0.654321,
                        "probability": 0.9,
                    }
                ],
            }
        ]
    data = {
        "source_wav": str(wav),
        "language": "ja",
        "segments": segments,
    }
    if extra:
        data.update(extra)
    (root / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    txt_path = root / "input.sc.txt"
    txt_path.write_text(txt, encoding="utf-8")
    return txt_path, wav


def _root(output: str) -> ET.Element:
    return ET.parse(output).getroot()


def _fraction_time(value: int, rate: int) -> str:
    f = Fraction(value, rate)
    return f"{f.numerator}s" if f.denominator == 1 else f"{f.numerator}/{f.denominator}s"


def _time_samples(value: str, rate: int) -> int:
    assert value.endswith("s")
    return int(Fraction(value[:-1]) * rate)


def test_default_and_custom_output_paths(tmp_path):
    txt, _ = _write_project(tmp_path)

    default = export_fcpxml(str(txt))
    custom_path = tmp_path / "custom.fcpxml"
    custom = export_fcpxml(str(txt), output=str(custom_path))

    assert default == str(tmp_path / "input.logic.fcpxml")
    assert Path(default).exists()
    assert custom == str(custom_path)
    assert custom_path.exists()


def test_fcpxml_structure_asset_uri_and_forbidden_elements(tmp_path):
    txt, wav = _write_project(tmp_path / "日本 語")
    output = export_fcpxml(str(txt))
    text = Path(output).read_text(encoding="utf-8")
    root = _root(output)

    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>')
    assert root.tag == "fcpxml"
    assert root.attrib["version"] == "1.8"
    resources = root.find("resources")
    assert resources is not None
    assert len(resources.findall("format")) == 1
    assets = resources.findall("asset")
    assert len(assets) == 1
    assert assets[0].attrib["src"] == wav.resolve().as_uri()
    assert root.find("project/sequence/spine") is not None
    assert root.find(".//transition") is None
    assert root.find(".//fadeIn") is None
    assert root.find(".//fadeOut") is None
    assert root.find(".//adjust-volume") is None
    assert "volume" not in text.lower()


def test_clip_times_are_sample_exact_with_reorder_gap_and_deterministic(tmp_path):
    segments = [
        {
            "id": "0001",
            "text": "A",
            "words": [],
            "kind": "unrecognized",
            "source_start": 0.123456,
            "source_end": 0.654321,
        },
        {
            "id": "0002",
            "text": "B",
            "words": [],
            "kind": "unrecognized",
            "source_start": 1.111111,
            "source_end": 1.999999,
        },
    ]
    txt, _ = _write_project(
        tmp_path,
        segments=segments,
        txt="[0002] B\n[0001] A\n",
    )
    output = export_fcpxml(str(txt))
    root = _root(output)
    clips = root.findall("project/sequence/spine/asset-clip")
    sequence = root.find("project/sequence")

    bounds = [(1.111111, 1.999999), (0.123456, 0.654321)]
    destination = 0
    for number, (clip, (start, end)) in enumerate(zip(clips, bounds), start=1):
        s0 = round(start * 44100)
        s1 = round(end * 44100)
        duration = s1 - s0
        assert clip.attrib["name"] == f"input {number:04d}"
        assert clip.attrib["start"] == _fraction_time(s0, 44100)
        assert clip.attrib["duration"] == _fraction_time(duration, 44100)
        assert clip.attrib["offset"] == _fraction_time(destination, 44100)
        assert clip.attrib["srcEnable"] == "audio"
        assert clip.attrib["audioRole"] == "dialogue"
        if number == 1:
            destination += 44100
        destination += duration
    assert len(clips) == 2
    assert sequence is not None
    assert sequence.attrib["duration"] == _fraction_time(destination, 44100)


def test_deleted_audio_becomes_same_length_timeline_gap(tmp_path):
    # 削除する 0002 は通常 speech 行にする（未認識行の削除は元位置ポーズの
    # guarded cut 判定が入るため、タイムラインギャップ生成の検証には使わない）。
    segments = [
        {"id": "0001", "text": "A", "words": [], "kind": "unrecognized", "source_start": 0.0, "source_end": 1.0},
        {"id": "0002", "text": "B",
         "words": [{"word": "B", "start": 1.0, "end": 2.0, "probability": 0.9}]},
        {"id": "0003", "text": "C", "words": [], "kind": "unrecognized", "source_start": 2.0, "source_end": 3.0},
    ]
    txt, _ = _write_project(tmp_path, segments=segments, txt="[0001] A\n[0003] C\n")

    root = _root(export_fcpxml(str(txt)))
    clips = root.findall("project/sequence/spine/asset-clip")
    sequence = root.find("project/sequence")

    assert [clip.attrib["offset"] for clip in clips] == ["0s", "2s"]
    assert sequence is not None
    assert sequence.attrib["duration"] == "3s"


def test_reordered_regions_have_one_second_boundary_gaps(tmp_path):
    segments = [
        {"id": "0001", "text": "A", "words": [], "kind": "unrecognized", "source_start": 0.0, "source_end": 1.0},
        {"id": "0002", "text": "B", "words": [], "kind": "unrecognized", "source_start": 1.0, "source_end": 2.0},
        {"id": "0003", "text": "C", "words": [], "kind": "unrecognized", "source_start": 2.0, "source_end": 3.0},
        {"id": "0004", "text": "D", "words": [], "kind": "unrecognized", "source_start": 3.0, "source_end": 4.0},
    ]
    txt, _ = _write_project(
        tmp_path,
        segments=segments,
        txt="[0001] A\n[0003] C\n[0002] B\n[0004] D\n",
    )

    root = _root(export_fcpxml(str(txt)))
    clips = root.findall("project/sequence/spine/asset-clip")
    sequence = root.find("project/sequence")

    assert [clip.attrib["offset"] for clip in clips] == ["0s", "2s", "4s", "6s"]
    assert sequence is not None
    assert sequence.attrib["duration"] == "7s"


@pytest.mark.parametrize(
    ("channels", "layout"), [(1, "mono"), (2, "stereo"), (3, "surround")]
)
def test_asset_and_sequence_audio_metadata(tmp_path, channels, layout):
    txt, _ = _write_project(tmp_path, channels=channels)
    root = _root(export_fcpxml(str(txt)))
    asset = root.find("resources/asset")
    sequence = root.find("project/sequence")

    assert asset is not None
    assert sequence is not None
    assert asset.attrib["audioRate"] == "44100"
    assert asset.attrib["audioChannels"] == str(channels)
    assert asset.attrib["duration"] == "6s"
    assert sequence.attrib["audioLayout"] == layout
    assert sequence.attrib["audioRate"] == "44.1k"


def test_unsupported_sequence_audio_rate_is_omitted(tmp_path):
    txt, _ = _write_project(tmp_path, samplerate=16000)
    sequence = _root(export_fcpxml(str(txt))).find("project/sequence")
    assert sequence is not None
    assert "audioRate" not in sequence.attrib


@pytest.mark.parametrize(
    "case", ["delete", "reorder", "filler", "long-silence", "guarded"]
)
def test_exporter_uses_shared_planned_intervals(tmp_path, case):
    if case == "guarded":
        segments = [
            {"id": "0001", "text": "あ",
             "words": [{"word": "あ", "start": 0.0, "end": 1.0}]},
            {
                "id": "0002",
                "text": "⚠ 未認識区間",
                "words": [],
                "kind": "unrecognized",
                "source_start": 1.2,
                "source_end": 1.6,
            },
            {"id": "0003", "text": "う",
             "words": [{"word": "う", "start": 1.8, "end": 2.8}]},
        ]
        txt_body = "[0001] あ\n[0003] う\n"
        extra = {"silences": [[1.0, 1.2], [1.6, 1.8]]}
    elif case == "reorder":
        segments = [
            {"id": "0001", "text": "あ", "words": [{"word": "あ", "start": 0.0, "end": 1.0}]},
            {"id": "0002", "text": "い", "words": [{"word": "い", "start": 1.0, "end": 2.0}]},
            {"id": "0003", "text": "う", "words": [{"word": "う", "start": 2.0, "end": 3.0}]},
        ]
        txt_body = "[0003] う\n[0001] あ\n[0002] い\n"
        extra = None
    elif case == "filler":
        segments = [{
            "id": "0001",
            "text": "あまあう",
            "words": [
                {"word": "あ", "start": 0.0, "end": 1.0},
                {"word": "まあ", "start": 1.2, "end": 2.0, "suggest": True},
                {"word": "う", "start": 2.2, "end": 3.0},
            ],
        }]
        txt_body = "[0001] あ〔まあ〕う\n"
        extra = {"silences": [[1.0, 1.2], [2.0, 2.2]]}
    elif case == "long-silence":
        segments = [
            {"id": "0001", "text": "あ", "words": [{"word": "あ", "start": 0.0, "end": 1.0}]},
            {"id": "0002", "text": "う", "words": [{"word": "う", "start": 4.0, "end": 5.0}]},
        ]
        txt_body = "[0001] あ\n[0002] う\n"
        extra = {"trim_silences": [[1.0, 4.0]]}
    else:
        segments = [{
            "id": "0001",
            "text": "あいう",
            "words": [
                {"word": "あ", "start": 0.0, "end": 1.0},
                {"word": "い", "start": 1.0, "end": 2.0},
                {"word": "う", "start": 2.0, "end": 3.0},
            ],
        }]
        txt_body = "[0001] あう\n"
        extra = None

    txt, _ = _write_project(
        tmp_path, segments=segments, txt=txt_body, extra=extra
    )
    plan = render.build_edit_plan(str(txt))
    output = export_fcpxml(str(txt))
    clips = _root(output).findall("project/sequence/spine/asset-clip")
    expected = []
    for start, end in plan.output_intervals:
        s0 = max(0, round(start * plan.samplerate))
        s1 = min(plan.total_samples, round(end * plan.samplerate))
        if s1 > s0:
            expected.append((s0, s1 - s0))

    actual = [
        (_time_samples(c.attrib["start"], 44100), _time_samples(c.attrib["duration"], 44100))
        for c in clips
    ]
    assert actual == expected


def test_silence_trim_preserves_fcpxml_timeline_gap(tmp_path):
    segments = [
        {
            "id": "0001",
            "text": "あ",
            "words": [{"word": "あ", "start": 0.0, "end": 1.0}],
        },
        {
            "id": "0002",
            "text": "う",
            "words": [{"word": "う", "start": 4.0, "end": 5.0}],
        },
    ]
    txt, _ = _write_project(
        tmp_path,
        segments=segments,
        txt="[0001] あ\n[0002] う\n",
        extra={"trim_silences": [[1.0, 4.0]]},
    )

    root = _root(export_fcpxml(str(txt)))
    clips = root.findall("project/sequence/spine/asset-clip")
    sequence = root.find("project/sequence")
    offsets = [_time_samples(clip.attrib["offset"], 44100) for clip in clips]
    starts = [_time_samples(clip.attrib["start"], 44100) for clip in clips]
    durations = [_time_samples(clip.attrib["duration"], 44100) for clip in clips]

    assert len(clips) == 2
    timeline_gap = offsets[1] - (offsets[0] + durations[0])
    source_gap = starts[1] - (starts[0] + durations[0])
    assert timeline_gap == source_gap
    assert timeline_gap > 0
    assert sequence is not None
    sequence_duration = _time_samples(sequence.attrib["duration"], 44100)
    assert sequence_duration == offsets[1] + durations[1]
    assert sequence_duration == starts[1] + durations[1]


def test_filler_cut_preserves_fcpxml_timeline_gap(tmp_path):
    segments = [{
        "id": "0001",
        "text": "あまあう",
        "words": [
            {"word": "あ", "start": 0.0, "end": 1.0},
            {"word": "まあ", "start": 1.2, "end": 2.0, "suggest": True},
            {"word": "う", "start": 2.2, "end": 3.0},
        ],
    }]
    txt, _ = _write_project(
        tmp_path,
        segments=segments,
        txt="[0001] あ〔まあ〕う\n",
        extra={"silences": [[1.0, 1.2], [2.0, 2.2]]},
    )

    clips = _root(
        export_fcpxml(str(txt))
    ).findall("project/sequence/spine/asset-clip")
    offsets = [_time_samples(clip.attrib["offset"], 44100) for clip in clips]
    starts = [_time_samples(clip.attrib["start"], 44100) for clip in clips]
    durations = [_time_samples(clip.attrib["duration"], 44100) for clip in clips]

    assert len(clips) == 2
    timeline_gap = offsets[1] - (offsets[0] + durations[0])
    source_gap = starts[1] - (starts[0] + durations[0])
    assert timeline_gap == source_gap
    assert timeline_gap > 0


def test_fcpxml_stats_duration_uses_logic_timeline_duration(tmp_path):
    segments = [
        {
            "id": "0001",
            "text": "あ",
            "words": [{"word": "あ", "start": 0.0, "end": 1.0}],
        },
        {
            "id": "0002",
            "text": "う",
            "words": [{"word": "う", "start": 4.0, "end": 5.0}],
        },
    ]
    txt, _ = _write_project(
        tmp_path,
        segments=segments,
        txt="[0001] あ\n[0002] う\n",
        extra={"trim_silences": [[1.0, 4.0]]},
    )
    stats: list[str] = []

    output = export_fcpxml(str(txt), stats_out=stats)
    sequence = _root(output).find("project/sequence")

    assert sequence is not None
    duration_s = _time_samples(sequence.attrib["duration"], 44100) / 44100
    expected = render.format_stats(
        render.EditStats(
            source_duration_s=6.0,
            output_duration_s=duration_s,
            filler_requested=0,
            filler_cut=0,
            filler_skipped_unsafe=0,
            silence_trim_count=1,
            silence_trim_removed_s=2.08,
            filler_cut_removed_s=0.0,
            lines_deleted_removed_s=0.0,
            lines_total=2,
            lines_deleted=0,
            lines_moved=0,
        ),
        gaps_preserved=True,
    )
    assert stats == expected


def test_planner_warning_is_printed_once(tmp_path, capsys):
    txt, _ = _write_project(tmp_path, txt="[0001] 音声追加\n")
    export_fcpxml(str(txt))
    err = capsys.readouterr().err
    assert err.count("警告:") == 1


def _write_sine_project(
    root: Path,
    *,
    freq: float = 240.0,
    channels: int = 1,
    samplerate: int = 48000,
    segments: list[dict] | None = None,
    txt: str = "[0001] A\n[0002] B\n",
) -> tuple[Path, Path, int]:
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    wav = root / "元 音声.wav"
    frames = samplerate * 6
    t = np.arange(frames) / samplerate
    mono = 0.5 * np.sin(2 * np.pi * freq * t)
    audio = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    sf.write(wav, audio, samplerate, subtype="PCM_16")
    if segments is None:
        segments = [
            {
                "id": "0001",
                "text": "A",
                "words": [],
                "kind": "unrecognized",
                "source_start": 0.5003,
                "source_end": 1.4007,
            },
            {
                "id": "0002",
                "text": "B",
                "words": [],
                "kind": "unrecognized",
                "source_start": 2.3001,
                "source_end": 3.2009,
            },
        ]
    data = {"source_wav": str(wav), "language": "ja", "segments": segments}
    (root / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    txt_path = root / "input.sc.txt"
    txt_path.write_text(txt, encoding="utf-8")
    return txt_path, wav, samplerate


def test_snap_edge_leaves_safe_boundary_unchanged():
    sr = 48000
    audio = np.zeros(sr)
    assert _snap_edge_inward(audio, sr, 1000, +1, sr - 1) == 1000
    assert _snap_edge_inward(audio, sr, 1000, -1, 0) == 1000


def test_snap_edge_moves_high_amplitude_edge_inward_to_quiet():
    sr = 48000
    t = np.arange(sr) / sr
    audio = 0.5 * np.sin(2 * np.pi * 240 * t)  # exact zeros every 100 samples
    max_search = int(round(EDGE_SEARCH_S * sr))
    # peak at sample 50 (phase pi/2) -> start edge scans forward
    start_new = _snap_edge_inward(audio, sr, 50, +1, len(audio) - 1)
    assert start_new >= 50
    assert start_new - 50 <= max_search
    assert abs(audio[start_new]) <= _SAFE_AMP
    # peak at sample 250 -> end edge scans backward
    end_new = _snap_edge_inward(audio, sr, 250, -1, 0)
    assert end_new <= 250
    assert 250 - end_new <= max_search
    assert abs(audio[end_new]) <= _SAFE_AMP


def test_snap_edge_respects_limit_sample():
    sr = 48000
    t = np.arange(sr) / sr
    audio = 0.5 * np.sin(2 * np.pi * 240 * t)
    # forward direction cannot pass limit_sample
    result = _snap_edge_inward(audio, sr, 50, +1, 70)
    assert 50 <= result <= 70


def test_snap_edge_stereo_uses_cross_channel_max():
    sr = 48000
    t = np.arange(sr) / sr
    left = 0.5 * np.sin(2 * np.pi * 220 * t)
    right = 0.5 * np.cos(2 * np.pi * 220 * t)  # never silent together
    audio = np.stack([left, right], axis=1)
    max_search = int(round(EDGE_SEARCH_S * sr))
    peak = 100
    result = _snap_edge_inward(audio, sr, peak, +1, len(audio) - 1)
    window = np.max(np.abs(audio[peak : peak + max_search + 1]), axis=1)
    expected = peak + int(np.argmin(window))
    # no sub-threshold point exists -> falls back to min cross-channel max
    assert window.min() > _SAFE_AMP
    assert result == expected


def test_snap_region_only_moves_inward():
    sr = 48000
    t = np.arange(sr) / sr
    audio = 0.5 * np.sin(2 * np.pi * 240 * t)
    start, end = 5013, 9007
    new_start, new_end = _snap_region(audio, sr, start, end)
    assert new_start >= start
    assert new_end <= end
    assert new_end - new_start >= 1
    assert abs(audio[new_start]) <= _SAFE_AMP
    assert abs(audio[new_end - 1]) <= _SAFE_AMP


@pytest.mark.parametrize("length", [1, 2, 5, 480])
def test_snap_region_keeps_minimum_length(length):
    sr = 48000
    t = np.arange(sr) / sr
    audio = 0.5 * np.sin(2 * np.pi * 240 * t)
    start = 1013
    end = start + length
    new_start, new_end = _snap_region(audio, sr, start, end)
    assert new_end - new_start >= 1
    assert new_start >= start
    assert new_end <= end


def test_export_snaps_clip_boundaries_to_quiet(tmp_path):
    import soundfile as sf

    txt, wav, sr = _write_sine_project(tmp_path)
    audio, _ = sf.read(wav, dtype="float64", always_2d=False)
    output = export_fcpxml(str(txt))
    clips = _root(output).findall("project/sequence/spine/asset-clip")
    assert clips
    for clip in clips:
        start = _time_samples(clip.attrib["start"], sr)
        duration = _time_samples(clip.attrib["duration"], sr)
        end = start + duration
        assert duration >= 1
        assert abs(audio[start]) <= _SAFE_AMP
        assert abs(audio[end - 1]) <= _SAFE_AMP


def test_generated_xml_validates_against_logic_dtd_when_available(tmp_path):
    dtd = Path("/Applications/Logic Pro.app/Contents/Resources/FinalCutProX_DTD_v1.8.dtd")
    xmllint = shutil.which("xmllint")
    if not dtd.exists() or xmllint is None:
        pytest.skip("Logic Pro FCPXML 1.8 DTD または xmllint がありません")
    txt, _ = _write_project(tmp_path, channels=2)
    output = export_fcpxml(str(txt))
    # xmllint は空白を含む --dtdvalid のパスを外部IDとして解決できないため、
    # Logic Pro同梱DTDを空白のない一時パスへコピーして同じ内容で検証する。
    validation_dtd = tmp_path / "FinalCutProX_DTD_v1.8.dtd"
    shutil.copyfile(dtd, validation_dtd)
    result = subprocess.run(
        [xmllint, "--noout", "--dtdvalid", str(validation_dtd), output],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_forward_overlap_trims_previous_end_and_keeps_source_timeline():
    # ep31: 隣接区間がソース上で 20ms 重なると、以降の全クリップが 20ms ずつ
    # 前へ詰まり、2人収録の別トラック間で時間軸がずれた。
    sr = 48000
    bounds = [(0, 1000), (900, 2000), (2500, 3000)]
    resolved = _resolve_forward_overlaps(bounds)
    assert resolved == [(0, 900), (900, 2000), (2500, 3000)]
    offsets, end = _timeline_offsets(resolved, sr)
    assert offsets == [start for start, _ in resolved]
    assert end == 3000


def test_resolve_forward_overlaps_leaves_reorder_and_containment_alone():
    reordered = [(1000, 2000), (0, 500)]
    assert _resolve_forward_overlaps(reordered) == reordered
    contained = [(0, 1000), (200, 800)]
    assert _resolve_forward_overlaps(contained) == contained
    same_start = [(0, 1000), (0, 1500)]
    assert _resolve_forward_overlaps(same_start) == same_start
