<p align="center">
  <img src="assets/logo-light.svg#gh-light-mode-only" alt="Stefnceorf" width="160" height="160">
  <img src="assets/logo-dark.svg#gh-dark-mode-only" alt="Stefnceorf" width="160" height="160">
</p>

<h1 align="center">Stefnceorf</h1>

<p align="center">
  <strong>Edit audio by editing text — local, free, and built for AI agents to customize.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black.svg" alt="Apple Silicon">
  <a href="README_ja.md">日本語</a>
</p>

---

Stefnceorf is a free, open-source CLI tool for **text-based podcast and audio editing**. Transcribe your audio, edit the transcript as a plain text file, and the edits are applied to the audio — delete a line to remove a segment, delete a word to cut it, rearrange lines to reorder the audio. All processing runs locally on Apple Silicon GPU. No cloud, no subscription, no data leaves your machine.

A local, free, privacy-first alternative to [Descript](https://www.descript.com/) for podcast post-production — with native Japanese support and a codebase designed for AI coding agents to fork and customize.

> **Requires Apple Silicon Mac** (M1/M2/M3/M4). Intel Macs are not supported. See [FAQ](#can-i-use-stefnceorf-on-windows-or-linux) for cross-platform options.

## Quick Start

```sh
sc episode.wav                   # transcribe + generate Logic Pro FCPXML in one step
```

Just pass a WAV file — Stefnceorf transcribes the audio and generates `episode.sc.txt`, `episode.sc.json`, and `episode.logic.fcpxml`. Open the FCPXML in Logic Pro and you're editing.

Want to clean up the transcript first? Edit the text, then re-export:

```sh
sc episode.wav                   # generates .sc.txt + .sc.json + .logic.fcpxml
$EDITOR episode.sc.txt           # edit the transcript (delete lines, words, filler)
sc logic episode.sc.txt          # re-generate FCPXML after edits
sc render episode.sc.txt         # or render directly to .edited.wav
```

## What Is Text-Based Audio Editing?

Instead of scrubbing through a waveform to find and cut sections, you work with a text transcript. The tool maintains a mapping between each word in the transcript and its position in the audio. When you delete text, the corresponding audio is removed. When you rearrange lines, the audio follows.

```
[0001 0:00] We ended up deciding that／if there's tedious work／just build a tool
[0002 0:12] 〔um〕and／remove words like that
[0003 0:25] I have◆plenty of things／I want to talk about
```

- **Delete a line** → that entire segment is removed from the audio
- **Delete words within a line** → those words are cut from the audio
- **Rearrange lines** → audio segments play in the new order
- **`〔...〕` filler suggestions** → keep the brackets to delete, remove brackets to keep
- **`／` pause boundaries** → safe cut points (partial deletion removes the whole block between pauses)
- **`◆` low-confidence markers** → words to double-check by listening

Editing is **non-destructive** — the original audio file is never modified, and you can re-render as many times as you want.

## Features

- **Fully local & free** — Transcription runs on Apple Silicon GPU via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). No cloud services, no API keys, no cost.
- **Text-based editing** — Edit a plain text file in your favorite editor. Delete text to delete audio.
- **Filler word detection** — Automatically suggests filler words (`um`, `uh`, `えー`, `まあ`) for removal. You review each suggestion and decide.
- **Pause-based block editing** — Cuts snap to natural pause boundaries, preventing unnatural audio joins.
- **Confirmed long silence trimming** — Detected silences over 1.2 seconds are shortened to 0.7 seconds for Whisper hallucination prevention and output; unknown or partially covered gaps are preserved.
- **Verbatim mode** — Transcribes filler words and hesitations that Whisper normally absorbs, enabling a filler-removal workflow.
- **Hallucination detection & rescue** — Automatically detects Whisper hallucinations, rescues the affected audio by re-transcribing with safe settings.
- **Quality preservation** — WAV render output matches original sample rate and bit depth. Equal-power crossfade at cut boundaries prevents click noise.
- **Japanese & English** — Full support for both languages with language-specific filler dictionaries.

## Stefnceorf vs Descript

| | Stefnceorf | Descript |
|---|---|---|
| **Price** | Free (MIT License) | $24+/month |
| **Audio processing** | Local (Apple Silicon GPU) | Cloud-based |
| **Privacy** | Audio never leaves your machine | Audio uploaded to servers |
| **Japanese support** | Native (transcription + filler detection) | Limited |
| **Filler removal** | Semi-automatic (suggest → you review) | Automatic |
| **Customizable by AI agents** | Yes — CLAUDE.md, AGENTS.md, full test suite | No |
| **Interface** | CLI + any text editor | GUI application |
| **Platform** | macOS (Apple Silicon) | macOS, Windows, Web |
| **Video editing** | Audio only | Audio + Video |

**Choose Stefnceorf when:** you want free, local, privacy-first audio editing, especially for Japanese content, and you're comfortable with a CLI. **Choose Descript when:** you need a GUI, video editing, or cross-platform support.

## Who Is This For?

- **Podcasters** — Cut filler words, remove false starts, tighten pacing. A 1-hour episode edit that takes 45 minutes with a waveform editor becomes a 10-minute text edit.
- **Interview editors** — Remove tangents and rearrange segments to improve narrative flow.
- **Lecture & talk producers** — Remove verbal tics and long pauses from recorded presentations.
- **Anyone recording spoken audio** — Get a transcript and clean up the audio simultaneously.

## Fork & Customize with AI Agents

Stefnceorf is designed to be **forked and customized** using AI coding agents like [Claude Code](https://docs.anthropic.com/en/docs/claude-code), Cursor, or GitHub Copilot.

The repository includes:

- **`CLAUDE.md`** — Project context and conventions for Claude Code
- **`AGENTS.md`** — Workflow instructions for AI coding agents
- **Comprehensive test suite** — Unit tests, CLI tests, and acceptance tests
- **Clean Python codebase** — ~2,500 lines across 4 modules, well-structured for AI modification

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

Both `stefnceorf` and `sc` (shorthand) commands are registered. Subcommands can be abbreviated: `sc trans` = `sc transcribe`. Passing a `.wav` file directly (without a subcommand) runs `auto`.

Dependencies: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), numpy, soundfile.

## Usage

### Auto (Default)

```sh
sc input.wav [--lang ja|en] [--no-verbatim] [--no-filler-suggest] [-o output.fcpxml]
```

Runs transcription and FCPXML export in one step. Equivalent to `sc transcribe` followed by `sc logic`. This is the default when you pass a WAV file without a subcommand — `sc input.wav` and `sc auto input.wav` are identical.

Generates three files:

- `input.sc.txt` — editable transcript
- `input.sc.json` — word-level timestamp data (do not edit)
- `input.logic.fcpxml` — Logic Pro FCPXML (from the unedited transcript)

After editing the transcript, re-run `sc logic input.sc.txt` to generate an updated FCPXML.

### Transcribe

```sh
sc transcribe input.wav [--lang ja|en] [--verbatim] [--no-filler-suggest] [--model MODEL] [--pause-threshold 0.15]
```

Generates two files:

- `input.sc.txt` — editable transcript (one line per segment, with timestamps and filler suggestions)
- `input.sc.json` — word-level timestamp and confidence data (do not edit)

Key options:

- `--lang` — Language (default: `ja`). Use `--lang en` for English.
- `--verbatim` — **Enabled by default.** Transcribes filler words that Whisper normally absorbs, using a slower but more accurate model. Its transcription history is reset about every 4–6 minutes to contain long-context hallucinations. Disable with `--no-verbatim`.
- `--no-filler-suggest` — Disables filler word detection.
- `--pause-threshold` — Minimum pause duration (seconds) for block boundaries (default: `0.15`). Set to `0` for word-level editing (legacy behavior).

### Edit

Open `input.sc.txt` in any text editor. The editing rules are described in the [What Is Text-Based Audio Editing?](#what-is-text-based-audio-editing) section above.

Key points:

- Only **deletion** is supported. Adding or rewriting text has no effect (the original audio for new words doesn't exist).
- Rearrangement is **line-level only** (reordering words within a line is not supported).
- A `⚠ 未認識区間 X.X秒（音声保持）` line represents original audio that could not be transcribed safely. Keep the line to keep that audio, delete the line to delete it, or move the line to move the audio.
- Suspected hallucinations are retried in this order: verbatim first, then safe mode. Safe mode may lose filler words, so an unrecognized warning is kept when neither retry is trustworthy.
- Filler audio is removed only when both cut boundaries pass the acoustic safety checks. Otherwise, the filler is left in place: natural audio continuity takes priority over the number of fillers removed.
- Unknown IDs or malformed lines cause an error — the tool never silently ignores problems.

### Render

```sh
sc render input.sc.txt [-o output.wav] [--gap-threshold 1.2] [--gap-max 0.7]
```

Produces `input.edited.wav` (or the path given by `-o`). Cuts are made from the **original WAV** — no quality degradation from multiple re-renders.

### Send to Logic Pro

```sh
sc logic input.sc.txt [-o output.fcpxml] [--gap-threshold 1.2] [--gap-max 0.7]
```

Produces `input.logic.fcpxml` by default. In Logic Pro, choose **File > Import > Final Cut Pro XML** and select that file. The FCPXML references the original WAV directly and preserves transcript deletions, line reordering, filler decisions, and long-silence handling. Normal deletions remain as gaps matching the deleted duration, while reordered-region boundaries have a one-second gap. No fades or crossfades are included; add fades in Logic as needed. If the original WAV is moved, Logic may ask you to relink it.

## Filler Dictionary

Filler words are detected by **exact match** against a dictionary file (one word per line):

- Japanese: `stefnceorf/fillers_ja.txt` (default: あのー, そのー, えー, えっと, えと, まあ, まー, うーん)
- English: `stefnceorf/fillers_en.txt` (default: um, uh, uhm, er, ah)

Edit these files to customize filler detection for your vocabulary.

## Frequently Asked Questions

### Is Stefnceorf really free?

Yes. It is MIT licensed. All processing runs on your local machine. There are no cloud services, API keys, or usage fees.

### How does Stefnceorf compare to Descript?

Stefnceorf is free and fully local — your audio never leaves your machine. The trade-off is that it's a CLI tool (no GUI) and requires an Apple Silicon Mac. Descript offers a GUI, video editing, and cross-platform support, but costs $24+/month and processes audio in the cloud. See the [comparison table](#stefnceorf-vs-descript) above.

### Does Stefnceorf work with English audio?

Yes. Use `--lang en` when transcribing. English filler dictionaries are included.

### How accurate is the transcription?

Japanese transcription accuracy is approximately 93–95%. English accuracy is comparable or better. Proper nouns and technical terms are the main sources of error. Low-confidence words are marked with `◆` for easy review.

### Can I use Stefnceorf on Windows or Linux?

Not directly — it requires Apple Silicon for mlx-whisper GPU acceleration. However, you can fork the repository and ask an AI coding agent to replace mlx-whisper with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for cross-platform support.

### Can I remove filler words automatically?

Filler removal is semi-automatic by design. The tool suggests filler words by wrapping them in `〔brackets〕`. You review each suggestion and decide whether to keep or remove it. This prevents accidental deletion of words like "that" or "so" that can be both filler and meaningful.

### What audio formats are supported?

Input must be WAV. Apply noise reduction beforehand (e.g., with Audacity). Stefnceorf handles silence trimming.

### Can AI coding agents modify this codebase?

Yes. The repository includes `CLAUDE.md` and `AGENTS.md` with project context and workflow instructions. The codebase is intentionally simple (~2,500 lines of Python) with comprehensive tests, making it well-suited for AI-assisted modification.

## Accuracy & Limitations

- Transcription accuracy is ~93–95% for Japanese. Proper nouns and jargon may vary.
- Word timestamp margins of ±20ms are compensated by crossfade, but **a final listen-through is always recommended**.
- `--verbatim` mode may trigger Whisper hallucinations in long silent sections. These are automatically detected and rescued, but check any warnings.
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

## Name

*Stefnceorf* — from Old English *stefn* (voice) + *ceorfan* (to cut).

## License

[MIT License](LICENSE)

---

📖 [日本語ドキュメント / Japanese documentation](README_ja.md)
