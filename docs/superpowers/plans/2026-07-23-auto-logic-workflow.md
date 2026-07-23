# Auto Logic Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run transcription and Logic Pro FCPXML export with `sc input.wav` or `sc auto input.wav`.

**Architecture:** Add an `auto` subcommand that composes the existing transcription and FCPXML functions. Normalize a leading WAV path to the `auto` command before argparse parsing, and extract the existing transcription dispatch/printing into a shared CLI helper so staged and automatic execution stay identical.

**Tech Stack:** Python 3, argparse, pytest

## Global Constraints

- Do not modify transcription, filler-cut, unrecognized-audio, render, or FCPXML planning semantics.
- Never modify the source WAV.
- Stop before FCPXML export when transcription fails.
- Preserve all existing subcommands and options.
- Do not include unrelated working-tree documentation changes in the implementation commit.
- Do not push.

---

### Task 1: Automatic transcription-to-Logic CLI workflow

**Files:**
- Modify: `stefnceorf/cli.py`
- Test: `tests/test_cli.py`
- Modify: `README.md` only if its pre-existing uncommitted changes can remain outside this task's staged commit; otherwise leave documentation for the existing documentation work.

**Interfaces:**
- Consumes: `transcribe(input_wav, *, lang, model, filler_suggest, pause_threshold, verbatim) -> dict`
- Consumes: `export_fcpxml(txt_path, *, output, gap_threshold, gap_max) -> str`
- Produces: `_normalize_argv(argv: list[str]) -> list[str]`
- Produces: `sc auto INPUT.wav` and `sc INPUT.wav`

- [ ] **Step 1: Write failing parser and normalization tests**

Add tests proving:

```python
def test_normalize_wav_input_to_auto():
    assert cli._normalize_argv(["episode.wav"]) == ["auto", "episode.wav"]


def test_parser_auto_combines_transcribe_and_logic_defaults():
    args = cli._build_parser().parse_args(["auto", "episode.wav"])
    assert args.command == "auto"
    assert args.input == "episode.wav"
    assert args.lang == "ja"
    assert args.verbatim is True
    assert args.filler_suggest is True
    assert args.pause_threshold == pytest.approx(transcribe_mod.PAUSE_THRESHOLD_S)
    assert args.gap_threshold == pytest.approx(GAP_THRESHOLD_S)
    assert args.gap_max == pytest.approx(GAP_MAX_S)
    assert args.output is None
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```sh
pytest -q tests/test_cli.py -k 'normalize_wav_input_to_auto or parser_auto'
```

Expected: failure because `_normalize_argv` and the `auto` parser do not exist.

- [ ] **Step 3: Add parser helpers and WAV-path normalization**

In `stefnceorf/cli.py`:

- Import `Path`.
- Extract reusable argument registration for transcription and Logic export options.
- Register `auto` with both option groups and `-o/--output` for the FCPXML path.
- Implement `_normalize_argv`; only a first argument whose case-insensitive suffix is `.wav` is rewritten to `['auto', ...]`. Known commands and unknown non-WAV commands are untouched.
- In `main`, obtain a concrete argument list, print help and return `0` when it is empty, then parse the normalized list.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```sh
pytest -q tests/test_cli.py -k 'parser or normalize'
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing orchestration tests**

Add tests that monkeypatch the two existing stage functions and prove:

```python
def test_main_direct_wav_transcribes_then_exports_logic(...):
    rc = cli.main([str(wav), "--lang", "en", "--gap-max", "0.5"])
    assert rc == 0
    assert events == [
        ("transcribe", str(wav)),
        ("logic", str(tmp_path / "x.sc.txt")),
    ]


def test_main_auto_stops_before_logic_when_transcription_fails(...):
    assert cli.main(["auto", "missing.wav"]) == 1
    assert logic_calls == []
```

Also assert transcription options reach `transcribe`, Logic options reach `export_fcpxml`, generated paths are printed, an exporter error returns `1`, and `main([])` prints help without invoking either stage.

- [ ] **Step 6: Run orchestration tests and verify RED**

Run:

```sh
pytest -q tests/test_cli.py -k 'direct_wav or main_auto or no_args'
```

Expected: failure because `auto` is not dispatched.

- [ ] **Step 7: Extract shared transcription execution and implement auto dispatch**

Extract the current transcription call, error handling, generated-path output, hallucination reporting, silence summary, and filler summary into a helper returning `(return_code, result_or_none)`. Use that helper from both `transcribe`/`trans` and `auto`.

For `auto`:

1. Run the shared transcription helper.
2. Return immediately on a non-zero result.
3. Pass `result['txt_path']`, `args.output`, `args.gap_threshold`, and `args.gap_max` to `export_fcpxml`.
4. Catch the same expected exceptions as the existing `logic` command, print `エラー: ...` to standard error, and return `1`.
5. Print the generated FCPXML path and return `0`.

- [ ] **Step 8: Run CLI and full automated tests**

Run:

```sh
pytest -q tests/test_cli.py
pytest -q
python -m compileall -q stefnceorf tests
git diff --check
```

Expected: all tests pass, compilation succeeds, and diff check is clean.

- [ ] **Step 9: Update user documentation without staging unrelated edits**

Document `sc input.wav` and `sc auto input.wav` as the automatic Logic workflow. If the existing README edits cannot be separated safely, do not stage README files in this task; report that documentation remains in the user's existing documentation change set.

- [ ] **Step 10: Commit the reviewed task**

Stage only the implementation files and any separable documentation hunk:

```sh
git add stefnceorf/cli.py tests/test_cli.py
git commit -m "feat: add automatic Logic workflow"
```
