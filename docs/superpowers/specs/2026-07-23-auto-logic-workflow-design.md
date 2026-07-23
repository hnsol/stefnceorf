# Auto Logic Workflow Design

## Goal

Make the normal Logic Pro workflow a single command while preserving the existing staged commands for advanced use.

## Interface

- `sc input.wav` runs the automatic workflow.
- `sc auto input.wav` is the explicit equivalent.
- `sc` with no input shows help and performs no work.
- Existing `transcribe`/`trans`, `render`, and `logic` commands remain compatible.
- The automatic workflow accepts the existing transcription and Logic export options.

## Workflow

1. Transcribe the WAV and create `.sc.txt` and `.sc.json` beside it.
2. Use the generated `.sc.txt` without rewriting it.
3. Export `.logic.fcpxml` through the existing Logic exporter.
4. Stop immediately if either stage fails and return a non-zero status.

The original WAV is never modified. Existing output collision behavior remains unchanged; this feature only orchestrates the existing stages.

## Safety

Bracketed filler suggestions remain deletion requests, but the existing acoustic safety check may retain unsafe cuts. Unrecognized sections remain explicit retained-audio entries. The automatic command must not bypass either mechanism.

## Output

Print every generated path and the transcription summary already available from the staged workflow. Logic-export warnings remain on standard error.

## Testing

CLI tests cover direct-WAV normalization, the explicit `auto` command, option forwarding, stage order, failure short-circuiting, and compatibility of existing commands.
