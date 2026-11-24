"""Logic Pro 向け FCPXML 出力。"""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .render import GAP_MAX_S, GAP_THRESHOLD_S, build_edit_plan, format_stats


# Logic 用 FCPXML はフェードを持たないため、リージョン端を内側の静音点へ
# 寄せてクリックノイズを防ぐ。最大補正幅と安全とみなす振幅閾値。
EDGE_SEARCH_S = 0.005
EDGE_SAFE_DBFS = -50.0


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


def _abs_max_at(audio, index: int) -> float:
    """指定サンプルの全チャンネル横断の最大 |振幅| を返す。"""
    value = audio[index]
    if audio.ndim == 1:
        return abs(float(value))
    return float(np.max(np.abs(value)))


def _snap_edge_inward(
    audio, samplerate: int, sample: int, direction: int, limit_sample: int
) -> int:
    """リージョン端を内側（direction 方向）へ最大 EDGE_SEARCH_S 走査し補正する。

    全チャンネルの |振幅| が閾値以下になる最短距離の位置を返す。無ければ
    走査範囲内でチャンネル横断の最大 |振幅| が最小のサンプルを返す。走査
    範囲が空なら sample をそのまま返す。外側（direction の逆）へは動かない。
    """
    total = len(audio)
    if total == 0:
        return sample
    max_search = int(round(EDGE_SEARCH_S * samplerate))
    threshold = 10 ** (EDGE_SAFE_DBFS / 20.0)

    candidates: list[int] = []
    for distance in range(0, max_search + 1):
        idx = sample + direction * distance
        if idx < 0 or idx >= total:
            break
        if direction > 0 and idx > limit_sample:
            break
        if direction < 0 and idx < limit_sample:
            break
        candidates.append(idx)
    if not candidates:
        return sample

    best_idx = candidates[0]
    best_amp: float | None = None
    for idx in candidates:  # 距離の近い順
        amp = _abs_max_at(audio, idx)
        if amp <= threshold:
            return idx
        if best_amp is None or amp < best_amp:
            best_amp = amp
            best_idx = idx
    return best_idx


def _snap_region(
    audio, samplerate: int, start_sample: int, end_sample: int
) -> tuple[int, int]:
    """リージョン両端を内側の静音点へ補正する。

    start 端は後方（+1）、end 端は前方（-1）へ寄せる。end_sample は排他境界
    なので評価対象は最終サンプル index end_sample-1。外側へは動かさず、補正で
    長さが 1 サンプル未満になる場合は元値を返す。
    """
    total = len(audio)
    if total == 0 or end_sample - start_sample < 1:
        return start_sample, end_sample

    # 互いの補正が交差しないよう区間中点で制約する。
    mid = (start_sample + end_sample) // 2
    start_limit = min(mid, end_sample - 1)
    new_start = _snap_edge_inward(
        audio, samplerate, start_sample, +1, start_limit
    )
    new_start = max(start_sample, min(new_start, end_sample - 1))

    end_limit = new_start
    new_end_boundary = _snap_edge_inward(
        audio, samplerate, end_sample - 1, -1, end_limit
    )
    new_end = new_end_boundary + 1
    new_end = min(end_sample, max(new_end, new_start + 1))

    if new_end - new_start < 1:
        return start_sample, end_sample
    return new_start, new_end


def _audio_layout(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return "surround"


def _resolve_forward_overlaps(
    bounds: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """順方向に重なる隣接区間の前側終端を、次区間の始端まで切り詰める。

    計画区間はクロスフェード用マージンのぶん数十ms 重なることがある。spine は
    クリップを重ねられないため、そのままだと次区間以降がすべて重なり分だけ前へ
    詰まり、ソース時間軸からずれる（別トラックと並べると位置が合わない）。
    切り詰めた2区間はソース上で連続するため、つなぎ目は無編集と同じ音になる。
    並べ替え（始端が逆行・同一）と包含（次区間が前区間内で終わる）は対象外。
    """
    resolved = list(bounds)
    for index in range(1, len(resolved)):
        previous_start, previous_end = resolved[index - 1]
        current_start, current_end = resolved[index]
        if previous_start < current_start < previous_end < current_end:
            resolved[index - 1] = (previous_start, current_start)
    return resolved


def _timeline_offsets(
    bounds: list[tuple[int, int]],
    samplerate: int,
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
    stats_out: list[str] | None = None,
) -> str:
    """編集計画を Logic Pro 用 FCPXML 1.8 として出力する。

    stats_out が渡された場合は format_stats の行をそこへ append するだけで
    print しない（呼び出し元が任意の位置で出力できる）。
    None の場合は警告出力の後に print する（後方互換）。
    """
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
            source_start, source_end = _snap_region(
                plan.audio, rate, source_start, source_end
            )
            bounds.append((source_start, source_end))
    bounds = _resolve_forward_overlaps(bounds)
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

    timeline_stats = replace(
        plan.stats, output_duration_s=timeline_end / rate if rate else 0.0
    )
    if stats_out is not None:
        stats_out.extend(format_stats(timeline_stats, gaps_preserved=True))
    else:
        for line in format_stats(timeline_stats, gaps_preserved=True):
            print(line)

    return str(out_path)
