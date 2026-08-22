# Architecture

Berangaria Agent has one primary runtime (`--gui` or `--background`) and one
diagnostic runtime (the loopback web UI). Both share the same conversation,
vision, transcription fallback and TTS clients.

## Voice pipeline

```text
microphone
   │
   ▼
WebRTC VAD ── no speech ──► keep listening
   │ utterance
   ▼
local faster-whisper
   │
   ├── no wake phrase ─────► discard locally
   │
   └── wake + request
          │
          ├──► one screen capture
          │
          ▼
      GPT-5.6 Luna via OpenRouter SSE
          │ reply text deltas
          ▼
      sentence boundary buffer
          │ completed spoken phrase
          ▼
      Fish HTTP PCM stream ───────────► speakers
```

If local Whisper cannot initialize, the fallback path uses Vosk only to verify
the wake phrase and sends the accepted audio utterance to OpenRouter STT. The
Vosk model is loaded lazily and downloaded only when that path is needed.

## Modules

| Module | Responsibility |
| --- | --- |
| `background.py` | Voice-loop orchestration, VAD, wake matching, screen capture and audio playback |
| `local_transcription.py` | CUDA runtime discovery and local faster-whisper inference |
| `transcription.py` | OpenRouter STT fallback and transcript sanitization |
| `chat.py` | OpenRouter requests, SSE text streaming and bounded RAM history |
| `fish.py` | Fish Audio HTTP synthesis and low-latency PCM streaming |
| `service.py` | Shared single-turn application service |
| `gui.py` | Visible Windows status/control window |
| `server.py` | Tokenized diagnostic HTTP UI on loopback |
| `config.py` | YAML/environment loading, validation and typed settings |
| `prompts.py` | Desktop-agent system prompt and screen trust boundary |

`config.example.yaml` is versioned. The user's `config.yaml`, `.env`, model
caches, CUDA DLL and logs are local runtime state and are ignored by Git.

## State and data boundaries

- Conversation history is bounded by `history_turns` and exists only in RAM.
- A screenshot is captured only after an accepted wake phrase.
- Ordinary speech is transcribed locally for wake matching and then discarded.
- The remote model receives accepted text plus one current screenshot.
- OpenRouter requests explicitly deny endpoints that may collect prompt data;
  optional per-request ZDR routing is controlled by configuration.
- Fish receives completed spoken phrases, not microphone audio, the owner request or screenshots.
- The primary voice path overlaps Luna generation with Fish synthesis. Only a
  fully completed model reply is committed to RAM history.
- The diagnostic HTTP server binds to `127.0.0.1` and protects mutation routes
  with a random per-process token and an origin check.

## Latency accounting

The background loop records these stages independently:

- trailing VAD silence after the owner stops speaking;
- wake matching and STT;
- screen capture/resize;
- model time to first text and full response generation;
- Fish time to first PCM chunk;
- end-of-speech to first audible response;
- playback and total turn duration.

Responsiveness should be evaluated primarily with `first_voice`; total duration
includes playing the entire spoken response and is not a response-start metric.
`first_voice` is recorded after the first PCM samples are written to the output
device, while `fish_first` measures arrival of the first network chunk. The GUI
shows p95 only after at least 20 completed turns.
The primary background runtime reuses OpenRouter and Fish HTTP clients across
turns so repeated requests can reuse established connections.

## Failure policy

- Local STT initialization/runtime failure can fall back to Vosk + OpenRouter
  when explicitly enabled in configuration.
- Empty/noise-only local transcription is discarded without a cloud request.
- Luna failures leave the agent listening for the next request.
- Fish failures preserve the generated text in the GUI and are logged.
- A confirmed turn that fails in cloud STT, Luna or Fish emits a local Windows
  error sound so the voice-first runtime does not fail silently.
- The GUI stop action cancels an active cloud/TTS task, and each Luna turn plus
  Fish first audio has a bounded deadline.
- The agent has no operating-system action tools; screen content is observation,
  never an instruction source.
