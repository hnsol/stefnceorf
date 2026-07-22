"""Logic Pro 向け FCPXML 出力。"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from .render import GAP_MAX_S, GAP_THRESHOLD_S, build_edit_plan


_SEQUENCE_AUDIO_RATES = {
    32000: "32k",
    44100: "44.1k",
    48000: "48k",
    88200: "88.2k",
    96000: "96k",
    176400: "176.4k",
    192000: "192k",
}
REORDER_GAP_S = 1


def _time(samples: int, samplerate: int) -> str:
    value = Fraction(samples, samplerate)
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"


def _audio_layout(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return "surround"


def _timeline_offsets(
    bounds: list[tuple[int, int]], samplerate: int
) -> tuple[list[int], int]:
    if not bounds:
        return [], 0
    offsets = [0]
    destination = bounds[0][1] - bounds[0][0]
    for index in range(1, len(bounds)):
        previous_start, previous_end = bounds[index - 1]
        current_start, current_end = bounds[index]
        source_gap_start = min(previous_end, current_start)
        source_gap_end = max(previous_end, current_start)
        retained_in_gap = any(
            other_index not in (index - 1, index)
            and other_start < source_gap_end
            and other_end > source_gap_start
            for other_index, (other_start, other_end) in enumerate(bounds)
        )
        reordered = current_start < previous_start or retained_in_gap
        gap = samplerate * REORDER_GAP_S if reordered else max(
            0, current_start - previous_end
        )
        destination += gap
        offsets.append(destination)
        destination += current_end - current_start
    return offsets, destination


def export_fcpxml(
    txt_path: str,
    output: str | None = None,
    gap_threshold: float = GAP_THRESHOLD_S,
    gap_max: float = GAP_MAX_S,
) -> str:
    """編集計画を Logic Pro 用 FCPXML 1.8 として出力する。"""
    plan = build_edit_plan(
        txt_path, gap_threshold=gap_threshold, gap_max=gap_max
    )
    rate = int(plan.samplerate)
    channels = int(plan.audio_info.channels)

    root = ET.Element("fcpxml", {"version": "1.8"})
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r1",
            "name": "FFVideoFormat1080p30",
            "frameDuration": "1/30s",
            "width": "1920",
            "height": "1080",
        },
    )
    ET.SubElement(
        resources,
        "asset",
        {
            "id": "r2",
            "name": plan.source_wav.name,
            "src": plan.source_wav.resolve().as_uri(),
            "start": "0s",
            "duration": _time(plan.total_samples, rate),
            "hasVideo": "0",
            "hasAudio": "1",
            "audioSources": "1",
            "audioChannels": str(channels),
            "audioRate": str(rate),
        },
    )

    project = ET.SubElement(root, "project", {"name": plan.base_name})
    sequence_attrs = {
        "format": "r1",
        "tcStart": "0s",
        "tcFormat": "NDF",
        "audioLayout": _audio_layout(channels),
    }
    sequence_rate = _SEQUENCE_AUDIO_RATES.get(rate)
    if sequence_rate is not None:
        sequence_attrs["audioRate"] = sequence_rate
    sequence = ET.SubElement(project, "sequence", sequence_attrs)
    spine = ET.SubElement(sequence, "spine")

    bounds = []
    for start, end in plan.output_intervals:
        source_start = max(0, int(round(start * rate)))
        source_end = min(plan.total_samples, int(round(end * rate)))
        if source_end > source_start:
            bounds.append((source_start, source_end))
    offsets, timeline_end = _timeline_offsets(bounds, rate)

    for clip_number, ((source_start, source_end), offset) in enumerate(
        zip(bounds, offsets), start=1
    ):
        ET.SubElement(
            spine,
            "asset-clip",
            {
                "ref": "r2",
                "name": f"{plan.base_name} {clip_number:04d}",
                "offset": _time(offset, rate),
                "start": _time(source_start, rate),
                "duration": _time(source_end - source_start, rate),
                "srcEnable": "audio",
                "audioRole": "dialogue",
            },
        )
    sequence.set("duration", _time(timeline_end, rate))

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n"
        f"{xml}\n"
    )

    if output is None:
        out_path = plan.txt_path.parent / f"{plan.base_name}.logic.fcpxml"
    else:
        out_path = Path(output)
    out_path.write_text(document, encoding="utf-8")

    for warning in plan.warnings:
        print(f"警告: {warning}", file=sys.stderr)
    return str(out_path)
