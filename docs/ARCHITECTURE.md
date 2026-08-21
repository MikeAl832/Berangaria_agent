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
      Luna via OpenRouter
          │ reply text
          ▼
      Fish PCM stream ─────► speakers
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
| `chat.py` | OpenRouter/Luna request, structured response and bounded RAM history |
| `fish.py` | Fish Audio synthesis and low-latency PCM streaming |
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
- Fish receives reply text, not microphone audio or screenshots.
- The diagnostic HTTP server binds to `127.0.0.1` and protects mutation routes
  with a random per-process token and an origin check.

## Latency accounting

The background loop records these stages independently:

- trailing VAD silence after the owner stops speaking;
- wake matching and STT;
- screen capture/resize;
- Luna response generation;
- Fish time to first PCM chunk;
- end-of-speech to first audible response;
- playback and total turn duration.

Responsiveness should be evaluated primarily with `first_voice`; total duration
includes playing the entire spoken response and is not a response-start metric.

## Failure policy

- Local STT initialization/runtime failure can fall back to Vosk + OpenRouter
  when explicitly enabled in configuration.
- Empty/noise-only local transcription is discarded without a cloud request.
- Luna failures leave the agent listening for the next request.
- Fish failures preserve the generated text in the GUI and are logged.
- The agent has no operating-system action tools; screen content is observation,
  never an instruction source.
