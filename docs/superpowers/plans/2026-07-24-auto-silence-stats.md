# Auto Silence Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `auto` のFCPXML開始を通知し、検出済み無音の安全な部分だけを実際に短縮して、その実績を0件時も生成後に表示する。

**Architecture:** `render.plan_output_intervals()` の純無音分岐を、ギャップ完全被覆判定から検出済み無音との共通部分単位の短縮へ変更する。CLIは既存の `stats_out` を維持し、`auto` のFCPXML開始通知と `format_stats()` の常時Silence行だけを追加する。

**Tech Stack:** Python 3.10+、pytest、既存の `stefnceorf.cli` / `stefnceorf.render`

## Global Constraints

- 検出済み無音の外側は削除しない。
- 文字起こし時の日本語 `無音切り詰め` 表示は変更しない。
- Logic/render終了時のSilence統計は実際の出力編集計画から集計する。
- 無音短縮0件時は `Silence: trimmed 0 gaps, 0.0s` と表示する。
- フィラー処理、通常削除、並べ替え、FCPXML形式は変更しない。
- 既存の未コミット変更には触れない。

---

## File Map

- `stefnceorf/render.py`: 検出済み無音との共通部分だけを短縮し、統計行を整形する。
- `tests/test_render.py`: 境界不一致、未知区間保持、0件統計の回帰テスト。
- `stefnceorf/cli.py`: `auto` がFCPXML生成へ移る時点の開始通知。
- `tests/test_cli.py`: 開始通知と生成後統計の内容・順序。

### Task 1: Safe partial-overlap silence trimming

**Files:**
- Modify: `stefnceorf/render.py:530-646`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `plan_output_intervals(intervals, word_spans, gap_threshold, gap_max, filler_spans, filler_pause_keep, silence_spans) -> tuple[list[tuple[float, float]], list[bool]]`
- Produces: 同じシグネチャのまま、検出済み無音との共通部分ごとに安全なカット境界を返す。

- [ ] **Step 1: Write failing partial-overlap tests**

`tests/test_render.py` の `plan_output_intervals` テスト群へ追加する。

```python
def test_plan_output_intervals_trims_only_known_silence_overlap():
    out, flags = render.plan_output_intervals(
        [(0.0, 1.0), (4.0, 5.0)],
        [(0.0, 1.0), (4.0, 5.0)],
        gap_threshold=1.2,
        gap_max=0.7,
        silence_spans=[(1.2, 3.8)],
    )
    assert out[0] == pytest.approx((0.0, 1.55))
    assert out[1] == pytest.approx((3.45, 5.0))
    assert flags == [False]


def test_plan_output_intervals_preserves_partial_unknown_gap_edges():
    out, _ = render.plan_output_intervals(
        [(0.0, 1.0), (4.0, 5.0)],
        [(0.0, 1.0), (4.0, 5.0)],
        gap_threshold=1.2,
        gap_max=0.7,
        silence_spans=[(1.2, 3.8)],
    )
    assert out[0][1] > 1.2
    assert out[1][0] < 3.8
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_render.py::test_plan_output_intervals_trims_only_known_silence_overlap \
  tests/test_render.py::test_plan_output_intervals_preserves_partial_unknown_gap_edges
```

Expected: FAIL because the current complete-coverage condition returns one merged interval.

- [ ] **Step 3: Implement overlap-only trimming**

In `plan_output_intervals()`, replace `_known_silence_covers()` with a helper that returns merged intersections with the current gap.

```python
def _known_silence_overlaps(start: float, end: float) -> list[tuple[float, float]]:
    if not silence_spans or end <= start:
        return []
    merged: list[list[float]] = []
    for span_start, span_end in sorted(
        (max(start, float(s)), min(end, float(e)))
        for s, e in silence_spans
        if float(e) > start and float(s) < end
    ):
        if span_end <= span_start:
            continue
        if merged and span_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span_end)
        else:
            merged.append([span_start, span_end])
    return [(span_start, span_end) for span_start, span_end in merged]
```

In pure-silence branch, keep the existing `g <= gap_threshold` behavior. For a longer gap, trim each known overlap longer than `gap_max`, while leaving unknown edges and short overlaps intact.

```python
        # 4. Pure silence gap
        if g <= gap_threshold or g <= gap_max:
            pending[1] = cur[1]
            continue

        trimmed = False
        for silence_start, silence_end in _known_silence_overlaps(pe, ns):
            silence_duration = silence_end - silence_start
            if silence_duration <= gap_max:
                continue
            pending[1] = silence_start + half
            out.append((pending[0], pending[1]))
            cf_flags.append(False)
            pending = [silence_end - half, cur[1]]
            trimmed = True
        if not trimmed:
            pending[1] = cur[1]
```

- [ ] **Step 4: Run focused and existing gap tests**

Run:

```bash
.venv/bin/pytest -q tests/test_render.py -k 'plan_output_intervals or gap_trim or stats_silence'
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add stefnceorf/render.py tests/test_render.py
git commit -m "fix: trim detected silence with timestamp drift"
```

### Task 2: Always show output silence stats

**Files:**
- Modify: `stefnceorf/render.py:782-822`
- Test: `tests/test_render.py:1572-1641`

**Interfaces:**
- Consumes: `format_stats(stats: EditStats) -> list[str]`
- Produces: 常に1つの `Silence:` 行を含む統計行リスト。

- [ ] **Step 1: Write the failing zero-stats test**

```python
def test_format_stats_silence_shown_when_zero():
    lines = render.format_stats(_stats())
    assert "Silence: trimmed 0 gaps, 0.0s" in lines
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_render.py::test_format_stats_silence_shown_when_zero
```

Expected: FAIL because the current formatter omits the Silence line.

- [ ] **Step 3: Make Silence formatting unconditional**

Replace the conditional Silence block in `format_stats()`.

```python
    gap_word = "gap" if stats.silence_trim_count == 1 else "gaps"
    removed = (
        f"-{stats.silence_trim_removed_s:.1f}s"
        if stats.silence_trim_removed_s > 0
        else "0.0s"
    )
    lines.append(
        f"Silence: trimmed {stats.silence_trim_count} {gap_word}, {removed}"
    )
```

Update the docstring to state that the Silence line is always present.

- [ ] **Step 4: Run formatter and render tests**

Run:

```bash
.venv/bin/pytest -q tests/test_render.py -k 'format_stats or stats_'
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add stefnceorf/render.py tests/test_render.py
git commit -m "feat: always show silence edit stats"
```

### Task 3: Announce the auto Logic stage

**Files:**
- Modify: `stefnceorf/cli.py:173-198`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `_ANNOUNCEMENTS["logic"]`
- Produces: `auto` の文字起こし完了後、FCPXML出力前に `logic: exporting FCPXML for Logic Pro: <txt_path>`。

- [ ] **Step 1: Update the auto output tests first**

In `test_main_direct_wav_transcribes_then_exports_logic()`, require:

```python
    assert capsys.readouterr().out == (
        f"auto: full pipeline: transcribe -> FCPXML: {wav}\n"
        f"生成: {txt}\n"
        f"生成: {json_path}\n"
        f"logic: exporting FCPXML for Logic Pro: {txt}\n"
        f"生成: {fcpxml}\n"
    )
```

In `test_main_auto_announces_and_prints_duration()`, add:

```python
    logic_idx = next(
        i for i, line in enumerate(lines)
        if line == f"logic: exporting FCPXML for Logic Pro: {txt}"
    )
    generated_fcpxml_idx = max(
        i for i, line in enumerate(lines) if line.startswith("生成:")
    )
    silence_idx = next(i for i, line in enumerate(lines) if line.startswith("Silence:"))
    assert logic_idx < generated_fcpxml_idx < silence_idx
```

In `test_main_auto_exporter_error_returns_one()`, require the Logic-stage line immediately before the exporter error:

```python
    assert captured.out == (
        "auto: full pipeline: transcribe -> FCPXML: x.wav\n"
        f"生成: {tmp_path / 'x.sc.txt'}\n"
        f"生成: {tmp_path / 'x.sc.json'}\n"
        "フィラー候補: 0箇所\n"
        f"logic: exporting FCPXML for Logic Pro: {tmp_path / 'x.sc.txt'}\n"
    )
```

- [ ] **Step 2: Run focused CLI tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_cli.py::test_main_direct_wav_transcribes_then_exports_logic \
  tests/test_cli.py::test_main_auto_announces_and_prints_duration
```

Expected: FAIL because the auto Logic-stage announcement is absent.

- [ ] **Step 3: Add the stage announcement**

Immediately before `export_fcpxml()` in the `auto` branch:

```python
        print(
            f"logic: {_ANNOUNCEMENTS['logic']}: {res['txt_path']}"
        )
```

- [ ] **Step 4: Run all CLI tests**

Run:

```bash
.venv/bin/pytest -q tests/test_cli.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add stefnceorf/cli.py tests/test_cli.py
git commit -m "feat: announce Logic stage in auto workflow"
```

### Task 4: Full verification with the real sample

**Files:**
- Verify only; do not modify sample files.

**Interfaces:**
- Consumes: `samples/260720_ep22-2.sc.txt` and its JSON/WAV.
- Produces: evidence that real detected silence is trimmed and final stats reflect the edit plan.

- [ ] **Step 1: Verify sample edit statistics without regenerating artifacts**

Run:

```bash
.venv/bin/python -c "from stefnceorf.render import build_edit_plan, format_stats; p=build_edit_plan('samples/260720_ep22-2.sc.txt'); print(p.stats); print(*format_stats(p.stats), sep='\n')"
```

Expected:

- `silence_trim_count` is greater than `0`.
- `silence_trim_removed_s` is greater than `0.0`.
- The printed `Silence:` line matches those values.
- `output_duration_s` is shorter than the previous 1490.567 seconds.

- [ ] **Step 2: Run the related suites**

Run:

```bash
.venv/bin/pytest -q tests/test_render.py tests/test_cli.py tests/test_fcpxml.py
```

Expected: PASS.

- [ ] **Step 3: Run all automated verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q stefnceorf tests
git diff --check
```

Expected: all tests pass, compilation succeeds, and no whitespace errors are reported.

- [ ] **Step 4: Inspect task-only diff**

Run:

```bash
git status --short
git diff HEAD~3 -- stefnceorf/render.py stefnceorf/cli.py tests/test_render.py tests/test_cli.py
```

Expected: only the approved behavior and tests changed in these four files; pre-existing unrelated changes remain untouched.
