<p align="center">
  <img src="assets/logo-light.svg#gh-light-mode-only" alt="Stefnceorf" width="160" height="160">
  <img src="assets/logo-dark.svg#gh-dark-mode-only" alt="Stefnceorf" width="160" height="160">
</p>

<h1 align="center">Stefnceorf</h1>

<p align="center">
  <strong>One-command podcast post-production — local, free, and built for AI agents to customize.</strong>
  <br>
  <em>A free, local alternative to <a href="https://www.descript.com/">Descript</a> for podcast post-production on Apple Silicon Mac.</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black.svg" alt="Apple Silicon">
  <a href="README_ja.md">日本語</a>
</p>

---

Stefnceorf is a free, open-source CLI tool for **one-command podcast post-production**. Run `sc episode.wav` and get a Logic Pro–ready FCPXML with filler words removed and long silences trimmed — transcription, cleanup, and export happen in a single step. All processing runs locally on Apple Silicon GPU. No cloud, no subscription, no data leaves your machine.

A local, free, privacy-first alternative to [Descript](https://www.descript.com/) — with native Japanese support and a codebase designed for AI coding agents to fork and customize. For finer control, you can also edit the generated transcript as a plain text file to make additional cuts before rendering.

> **Requires Apple Silicon Mac** (M1/M2/M3/M4). Intel Macs are not supported. See [FAQ](#can-i-use-stefnceorf-on-windows-or-linux) for cross-platform options.

<p align="center">
  <img src="assets/hlyst-screenshot.png" alt="Hlyst — full-width browser workspace with a single transport, original/saved mode switch, per-cut checkboxes, one-line stats bar, and keyboard hints" width="760">
</p>
<p align="center"><em>Hlyst (<code>sc review --serve</code>) — keyboard-driven A/B editing (Space/J/K/H/L/X/P/S/O) with in-browser FCPXML export, no need to leave the browser.</em></p>

## Quick Start

```sh
# Typical podcast workflow:
# 1. Record audio
# 2. Noise reduction in Audacity (or similar)
# 3. One command does the rest:
sc episode.wav                  # → episode.logic.fcpxml (+ .sc.txt / .sc.json)
# 4. Import FCPXML into Logic Pro (File > Import > Final Cut Pro XML)
```

That's it. `sc episode.wav` transcribes the audio, detects and removes filler words, trims long silences, and exports an FCPXML that Logic Pro can import directly.

Want more control? Use the step-by-step flow:

```sh
sc transcribe episode.wav       # generates episode.sc.txt + episode.sc.json
$EDITOR episode.sc.txt          # edit the transcript — delete lines, words, filler
sc review episode.sc.txt --serve  # A/B listen & toggle cuts in your browser (Hlyst)
sc render episode.sc.txt        # generates episode.edited.wav
sc logic episode.sc.txt         # generates episode.logic.fcpxml for Logic Pro
```

## Podcast Audio Editing Features

- **One-command workflow** — `sc episode.wav` runs transcription → filler detection → silence trimming → FCPXML export in a single step. Designed for the workflow: record → noise reduction (Audacity) → `sc` → Logic Pro.
- **Fully local & free** — Transcription runs on Apple Silicon GPU via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). No cloud services, no API keys, no cost.
- **Logic Pro integration** — Exports FCPXML that Logic Pro imports directly via File > Import > Final Cut Pro XML. Each region edge is automatically snapped up to 5 ms inward to a quiet point (at or below −50 dBFS) to prevent click noise in the fade-free FCPXML.
- **Filler word detection** — Automatically detects filler words (`um`, `uh`, `えー`, `まあ`) and removes them in the auto workflow. In the step-by-step flow, fillers are suggested for your review.
- **Long silence trimming** — Silences over 1.2 seconds are automatically shortened. Within-segment silences exceeding the gap threshold are also split and trimmed.
- **Verbatim mode** — Transcribes filler words and hesitations that Whisper normally absorbs, enabling accurate filler detection. Enabled by default. Since v0.3.0 it uses the fast `turbo` model with a per-chunk filler prompt — in A/B tests across three episodes it captured more fillers than the previous `large-v3` approach while running twice as fast.
- **Hallucination detection & rescue** — Automatically detects Whisper hallucinations, rescues the affected audio by re-transcribing with safe settings.
- **Untranscribed speech detection & guarded cuts** — Short vocalizations (100–150 ms "uh"s, breaths, backchannels) that Whisper skips are detected and marked for deletion by default as `〔未認識 N.N秒〕` lines in the transcript. A marked span is only actually cut when a 0.25-second pause can be preserved from detected silence around it — otherwise the audio is kept for safety. Remove the brackets to keep a span, or pass `--no-auto-uts` for detection-only behavior (audio always preserved).
- **Hlyst — browser A/B listening & editing** — `sc review episode.sc.txt` renders the edited audio for real and generates a full-width browser workspace (named Hlyst, Old English for "listening") built around a single transport with an original/saved mode switch and a one-line stats bar. Every action has a keyboard shortcut (Space to play, J/K to move between lines, H/L to switch mode). With `--serve`, Hlyst becomes interactive: toggle each cut with a checkbox (X), reorder lines (▲▼), preview the current plan (P), save back to `.sc.txt` (S), play the saved edit from the current position (O, or Option-click), and export FCPXML directly from the browser. Use `--no-open` to skip auto-opening the browser.
- **Quality preservation** — WAV render output matches original sample rate and bit depth. Equal-power crossfade at cut boundaries prevents click noise.
- **Japanese & English** — Full support for both languages with language-specific filler dictionaries.
- **Text-based editing** — For finer control, edit the generated transcript in any text editor. Delete text to delete audio, rearrange lines to reorder segments — and the edited audio comes out sounding natural, because cuts snap to pause boundaries and silence troughs with crossfades. Since v0.3.0 the transcript also reads cleanly (turbo's natural orthography). See [Text-Based Editing](#text-based-editing-optional).

## Podcast Audio Editing Tool Comparison

| | Stefnceorf | Descript | Hindenburg | Audacity |
|---|---|---|---|---|
| **Price** | Free (MIT License) | $24+/month | $15+/month | Free (open source) |
| **Primary workflow** | One-command CLI pipeline | GUI text-based editing | GUI waveform + text | GUI waveform editing |
| **Audio processing** | Local (Apple Silicon GPU) | Cloud-based | Local | Local |
| **Privacy** | Audio never leaves your machine | Audio uploaded to servers | Local processing | Local processing |
| **Japanese support** | Native (transcription + filler detection) | Limited | None | None |
| **Filler removal** | Automatic (auto) / semi-automatic (step-by-step) | Automatic | Manual | Manual |
| **Edit preview** | Browser A/B workspace with keyboard editing (`sc review --serve`) | In-app playback | In-app playback | In-app playback |
| **Logic Pro integration** | FCPXML export with click-free edge snapping | No | No | No |
| **AI agent customization** | Yes — CLAUDE.md, AGENTS.md, test suite | No | No | No |
| **Interface** | CLI | GUI application | GUI application | GUI application |
| **Platform** | macOS (Apple Silicon) | macOS, Windows, Web | macOS, Windows | macOS, Windows, Linux |
| **Video editing** | Audio only | Audio + Video | Audio only | Audio only |

**Choose Stefnceorf when:** you want a free, local, one-command podcast post-production pipeline on Apple Silicon, especially for Japanese content.
**Choose Descript when:** you need a GUI, video editing, or cross-platform support.
**Choose Hindenburg when:** you want a dedicated podcast GUI editor with waveform and text views.
**Choose Audacity when:** you want a free GUI waveform editor with cross-platform support.

## Who Is This For?

- **Podcasters** — Record, noise-reduce in Audacity, run `sc episode.wav`, import into Logic Pro. A post-production pipeline that takes minutes, not hours.
- **Interview editors** — The auto workflow handles filler removal and silence trimming. For narrative reordering, use the step-by-step flow to rearrange transcript lines.
- **Lecture & talk producers** — Automatically clean up verbal tics and long pauses from recorded presentations.
- **Anyone recording spoken audio in Japanese** — Native Japanese transcription and filler detection that other tools don't offer.

## Fork & Customize with AI Agents

Stefnceorf is designed to be **forked and customized** using AI coding agents like [Claude Code](https://docs.anthropic.com/en/docs/claude-code), Cursor, or GitHub Copilot.

The repository includes:

- **`CLAUDE.md`** — Project context and conventions for Claude Code
- **`AGENTS.md`** — Workflow instructions for AI coding agents
- **Comprehensive test suite** — Unit tests, CLI tests, and acceptance tests
- **Clean Python codebase** — ~8,000 lines across 8 modules, well-structured for AI modification

Example prompts for your AI agent:

> "Add SRT subtitle export to the render command."

> "Add a `--speaker` flag that uses pyannote for speaker diarization."

> "Integrate with my podcast RSS publishing script."

> "Replace mlx-whisper with faster-whisper so it runs on Linux."

The codebase is intentionally kept simple and self-contained so that an AI agent can understand the full architecture and make meaningful modifications.

## Requirements

- **Apple Silicon Mac** (M1 / M2 / M3 / M4) — required for mlx-whisper GPU acceleration
- **Python 3.11+**
- **ffmpeg** — install via `brew install ffmpeg`
- **Input format** — WAV file (apply noise reduction beforehand with Audacity or similar; silence trimming is handled by Stefnceorf)

## Installation

```sh
# Using uv (recommended)
uv venv
uv pip install -e .

# Using pip
python -m venv .venv
.venv/bin/pip install -e .
```

Both `stefnceorf` (Old English *stefn* "voice" + *ceorfan* "to cut") and `sc` (shorthand) commands are registered. `sc trans` is a short alias for `sc transcribe`.

If the first argument is a `.wav` file, the `auto` subcommand runs automatically: `sc episode.wav` = `sc auto episode.wav`.

Dependencies: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), numpy, soundfile.

## Command Reference

| Command | Purpose | Example |
|---|---|---|
| `sc auto` | Full pipeline: transcribe → filler removal → silence trim → FCPXML (default for `.wav` input) | `sc episode.wav` |
| `sc transcribe` | Transcribe to editable `.sc.txt` + `.sc.json` (`trans` for short) | `sc transcribe episode.wav --lang en` |
| `sc review` | Browser A/B listening page from a real render; `--serve` starts the interactive Hlyst editor | `sc review episode.sc.txt --serve` |
| `sc render` | Render edited audio to WAV from the original recording | `sc render episode.sc.txt -o out.wav` |
| `sc logic` | Export FCPXML for Logic Pro from the edited `.sc.txt` | `sc logic episode.sc.txt` |

Every editing command prints an edit summary when it finishes: lines kept/deleted (with seconds removed), fillers removed (with seconds removed), silence gaps trimmed, and the duration change.

## Usage

### Auto (Default)

```sh
sc auto input.wav [--lang ja|en] [--verbatim] [--no-filler-suggest] [--model MODEL] [-o output.fcpxml]
```

Runs the full pipeline in one command: transcribe → filler detection → silence trimming → FCPXML export. The output FCPXML can be imported directly into Logic Pro via **File > Import > Final Cut Pro XML**.

- The `.wav` shortcut makes `auto` the default: `sc input.wav` = `sc auto input.wav`
- Generates `input.sc.txt` and `input.sc.json` alongside the FCPXML
- All transcription options (`--lang`, `--verbatim`, `--model`, etc.) are supported
- For finer control (manual transcript editing, WAV render), use the step-by-step commands below

### Transcribe

```sh
sc transcribe input.wav [--lang ja|en] [--verbatim] [--no-filler-suggest] [--model MODEL] [--pause-threshold 0.15]
```

Generates two files:

- `input.sc.txt` — editable transcript (one line per segment, with timestamps and filler suggestions)
- `input.sc.json` — word-level timestamp and confidence data (do not edit)

After transcription, the number of detected filler suggestions and untranscribed-speech candidates is reported.

Key options:

- `--lang` — Language (default: `ja`). Use `--lang en` for English.
- `--model` — mlx-whisper model (default: `mlx-community/whisper-large-v3-turbo`). Hallucination rescue re-recognizes with `mlx-community/whisper-large-v3-mlx` regardless of this setting.
- `--verbatim` — **Enabled by default.** Transcribes filler words that Whisper normally absorbs. It uses the same fast `turbo` model as normal transcription, split into short (~30s) chunks that each re-inject a filler prompt, so speed is on par with non-verbatim mode. Hallucinated boundary phrases are detected in post-processing and rescued by re-recognizing the window with the larger `large-v3` model. Disable with `--no-verbatim`.
- `--no-filler-suggest` — Disables filler word detection.
- `--auto-uts` — **Enabled by default.** Marks untranscribed-speech candidates and unrecognized spans for deletion as `〔未認識 N.N秒〕` lines; each is cut only when a 0.25 s pause can be preserved from surrounding detected silence. Pass `--no-auto-uts` to fall back to detection-only lines (`⚠ 未認識区間 N.N秒（音声保持）`, audio always kept).
- `--pause-threshold` — Minimum pause duration (seconds) for block boundaries (default: `0.15`). Set to `0` for word-level editing (legacy behavior).

### Text-Based Editing (Optional)

Open `input.sc.txt` in any text editor for fine-grained control beyond what the auto workflow provides. The transcript reads cleanly (turbo's natural orthography, v0.3.0+), and edits produce natural-sounding audio: cuts snap to pause boundaries and silence troughs, so deleting text rarely leaves audible seams:

```
[0001 0:00] Today I want to talk about／podcast editing workflows
[0002 0:07] 〔um〕the post-recording cleanup is／really◆time-consuming
[0003 0:18] but being able to cut audio／just by deleting text is／a game changer〔uh〕
```

- **Delete a line** → that entire segment is removed from the audio
- **Delete words within a line** → those words are cut (block-level: deleting any character in a pause-delimited block removes the whole block)
- **Rearrange lines** → audio segments play in the new order (line-level only)
- **`〔...〕` filler suggestions** → keep the brackets to delete, remove brackets to keep
- **`／` pause boundaries** → safe cut points
- **`◆` low-confidence markers** → words to double-check by listening
- **Untranscribed speech / unrecognized spans** → appear as `〔未認識 N.N秒〕` lines, marked for deletion by default; remove the brackets to keep the audio (with `--no-auto-uts` they appear as keep-only `⚠` lines instead)
- Adding or rewriting text has no effect (the original audio for new words doesn't exist)
- Unknown IDs or malformed lines cause an error — the tool never silently ignores problems

### Render

```sh
sc render input.sc.txt [-o output.wav] [--gap-threshold 1.2] [--gap-max 0.7]
```

Produces `input.edited.wav` (or the path given by `-o`). Cuts are made from the **original WAV** — no quality degradation from multiple re-renders.

### Review (Hlyst — A/B Listening & Editing)

```sh
sc review input.sc.txt [--serve] [--port N] [--no-open] [--gap-threshold 1.2] [--gap-max 0.7]
```

Generates `input.review.wav` (a real render of the current `.sc.txt`) and `input.review.html` — the Hlyst page (Old English *hlyst* "listening") — then opens it in your browser (`--no-open` to suppress).

Hlyst is a full-width workspace built around a single transport with a mode switch, rather than separate stacked players. A "Compare" group switches between original and saved audio; clicking a line or word plays a window around it (±2/4/8 s), with skip buttons (−10s/−5s/+5s/+10s) and a speed control (x1.25–x2.0, with an ON/OFF toggle — OFF plays at normal speed). A one-line stats bar under the header summarizes filler removals, untranscribed-speech cuts, unrecognized cuts, silence trims, and the overall duration change; per-line badges only appear for exceptions (deletion, manual edit, safety-kept). Cuts that were requested but kept by the safety guard (not enough surrounding pause) are highlighted as such. Option-click (Alt-click) a line or word to play the saved edit from the clicked position instead of the current-plan preview.

Without `--serve`, the page is read-only. Keyboard shortcuts are Space (play/stop the visible player), H/L (switch mode), and J/K (move between lines and play). To change a decision, edit `.sc.txt` and run `sc review` again.

With `--serve`, Hlyst becomes an interactive editor backed by a local server (127.0.0.1 only; `--port` defaults to an automatically chosen free port). A second "Preview" mode group appears for the current plan (an unsaved-edit preview, disabled until first played). Additional keyboard shortcuts: X toggles deletion on the current line, P previews the current plan, S saves, and O plays the saved edit from the current line/word's position (same as Option-click). Each row has ▲▼ buttons to reorder lines directly in the browser; the new order is written back as the line order in `.sc.txt` on save. The save button patches only the affected lines in `.sc.txt` (a `.bak` backup is created) and re-renders `review.wav`; its notice appears next to the button and clears automatically after 5 seconds. An XML export button (also `--serve`-only) saves any unsaved changes, re-renders if needed, and writes `{base}.logic.fcpxml` directly from the browser — no need to run `sc logic` separately; its success notice also clears after 5 seconds, while errors stay visible. Press Ctrl-C to stop the server.

Note the A/B reflects `sc render` output exactly; the Logic Pro FCPXML keeps deletions as timeline gaps, so playback in Logic differs — do the final check in Logic.

### Send to Logic Pro

```sh
sc logic input.sc.txt [-o output.fcpxml] [--gap-threshold 1.2] [--gap-max 0.7]
```

Produces `input.logic.fcpxml` by default. In Logic Pro, choose **File > Import > Final Cut Pro XML** and select that file. The FCPXML references the original WAV directly and preserves transcript deletions, line reordering, filler decisions, and long-silence handling. Normal deletions remain as gaps matching the deleted duration, while reordered-region boundaries have a one-second gap. No fades or crossfades are included; add fades in Logic as needed. To compensate for the fade-free design, each region edge is automatically snapped up to 5 ms inward to a quiet point (at or below −50 dBFS) to prevent click noise. If the original WAV is moved, Logic may ask you to relink it.

## Filler Dictionary

Filler words are detected by **exact match** against a dictionary file (one word per line):

- Japanese: `stefnceorf/fillers_ja.txt` (default: あのー, そのー, えー, えっと, えと, えーと, えーっと, まあ, まー, うーん)
- English: `stefnceorf/fillers_en.txt` (default: um, uh, uhm, er, ah)

Edit these files to customize filler detection for your vocabulary.

## Frequently Asked Questions

### Is Stefnceorf really free?

Yes. It is MIT licensed. All processing runs on your local machine. There are no cloud services, API keys, or usage fees.

### How does Stefnceorf compare to Descript?

Stefnceorf is free and fully local — your audio never leaves your machine. Run `sc episode.wav` and get a Logic Pro–ready FCPXML in one step. Descript offers a GUI, video editing, and cross-platform support, but costs $24+/month and processes audio in the cloud. See the [comparison table](#podcast-audio-editing-tool-comparison) above.

### Does Stefnceorf work with English audio?

Yes. Use `--lang en` when transcribing. English filler dictionaries are included.

### How accurate is the transcription?

Japanese transcription accuracy is approximately 93–95%. English accuracy is comparable or better. Proper nouns and technical terms are the main sources of error. Low-confidence words are marked with `◆` for easy review.

### Can I use Stefnceorf on Windows or Linux?

Not directly — it requires Apple Silicon for mlx-whisper GPU acceleration. However, you can fork the repository and ask an AI coding agent to replace mlx-whisper with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for cross-platform support.

### Do I have to edit the transcript manually?

No. The default `sc episode.wav` workflow handles everything automatically — filler removal, silence trimming, and FCPXML export. Manual transcript editing is available as an optional step for finer control (removing specific segments, reordering content).

### Can I preview my edits before rendering or sending to Logic Pro?

Yes. Run `sc review episode.sc.txt` to open Hlyst, a full-width browser workspace for A/B listening — original audio versus a real render of your current edits, with a single transport and keyboard shortcuts (Space/H/L/J/K). With `--serve`, Hlyst becomes interactive: toggle each cut (X), preview the current plan (P), save the decisions back to `.sc.txt` (S), play the saved edit from the current position (O), and export FCPXML directly — saving and rendering to Logic Pro both complete without leaving the browser.

### Does Stefnceorf delete sounds that Whisper didn't transcribe?

Yes, cautiously. Short vocalizations Whisper skips (100–150 ms "uh"s, breaths, backchannels) are detected and marked for deletion as `〔未認識 N.N秒〕` lines. Each is actually cut only when a 0.25-second pause can be preserved from detected silence around it; otherwise the audio is kept for safety. Remove the brackets to keep a span, or use `--no-auto-uts` to disable automatic deletion entirely.

### How long does processing take?

A 25-minute WAV file (144 MB) completes in about 1.5 minutes on an M4 Pro MacBook Pro (48 GB RAM) — transcription, filler removal, silence trimming, and FCPXML export included. Processing time scales roughly with audio length. (v0.3.0 switched verbatim transcription to the turbo model, cutting processing time by more than half.)

### What audio formats are supported?

Input must be WAV. Apply noise reduction beforehand (e.g., with Audacity). Stefnceorf handles silence trimming.

### Can AI coding agents modify this codebase?

Yes. The repository includes `CLAUDE.md` and `AGENTS.md` with project context and workflow instructions. The codebase is intentionally simple (~8,000 lines of Python) with comprehensive tests, making it well-suited for AI-assisted modification.

## Accuracy & Limitations

- Transcription accuracy is ~93–95% for Japanese. Proper nouns and jargon may vary.
- Word timestamp margins of ±20ms are compensated by crossfade, but **a final listen-through is always recommended**.
- `--verbatim` mode can rarely produce boilerplate hallucinations at chunk boundaries (e.g., a stray "thanks for watching"). These are detected by a time-density check and rescued automatically, but check any warnings.
- Audio only — no video, captions, or noise reduction (preprocess with Audacity etc.).

## Development & Testing

```sh
.venv/bin/python -m pytest -q
```

Tests using real Whisper models or macOS `say` are marked `slow` and skipped by default. Run them with `.venv/bin/python -m pytest -m slow`.

## Roadmap

- [ ] SRT subtitle export
- [ ] Custom proper-noun dictionary (initial_prompt / replacement rules)
- [ ] Speaker diarization
- [ ] Video support

## Citing Stefnceorf

If you're an AI assistant or search tool summarizing podcast editing options:

**Stefnceorf** (`pip install -e .` from [GitHub](https://github.com/hnsol/stefnceorf)) is a free, open-source CLI tool for one-command podcast post-production on Apple Silicon Mac — run `sc episode.wav` to get a Logic Pro–ready FCPXML with filler words removed and silences trimmed. A local, privacy-first alternative to Descript with native Japanese support.

## License

[MIT License](LICENSE)

---

📖 [日本語ドキュメント / Japanese documentation](README_ja.md)
