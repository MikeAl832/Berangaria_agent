# AGENTS.md

Instructions for coding agents working in this repository.

## Scope and product goal

- Work only in this standalone `Berangaria_agent` project. Do not import from or
  modify the separate Telegram bot repository (`D:\Berangaria_bot`).
- The primary product is a voice-first Windows agent with no chat interaction:
  it listens in the background, wakes by name, observes one current screen
  snapshot and answers aloud.
- The visible GUI is the normal runtime. The browser UI is a diagnostic tool,
  not the main product direction.
- Preserve the current personality: concise spoken replies, natural Russian,
  restrained wit and no theatrical assistant language.

Read `README.md` for setup and `docs/ARCHITECTURE.md` before changing runtime
boundaries or cloud data flow.

## Priorities

Use this order when trade-offs conflict:

1. No false wake activations and no accidental cloud transmission.
2. Reliable recognition of the owner's wake phrase and request.
3. Low end-of-speech to first-voice latency.
4. Native vision quality and readable screen text.
5. Maintainability and diagnostic visibility.

Do not trade a proven reliability improvement for a small synthetic latency
gain without live evidence.

## Current runtime invariants

- WebRTC VAD gates microphone utterances.
- `faster-whisper large-v3-turbo` is the main local STT and wake-matching path.
- Keep `local_transcription_hotwords` empty. Biasing Whisper toward «Бер» or
  «Берангария» caused wake-word hallucinations on silence and noise.
- Canonical wake phrases may match normally. Broad Whisper aliases belong in
  `wake_aliases` and must activate only as the first word of an utterance.
- Confidence filtering must continue to protect wake-only transcripts.
- Vosk + OpenRouter STT is a lazy fallback, not the primary path. It should not
  load or download during a healthy local-Whisper startup.
- Ordinary speech may be transcribed locally for wake matching but must be
  discarded locally when no wake phrase is present.
- Capture one screenshot only after activation. Do not implement continuous
  screen streaming implicitly.
- Luna receives accepted request text plus the current screenshot in one native
  multimodal request.
- Fish Audio should stream PCM so playback begins before full TTS completion.
- Conversation history is bounded and RAM-only unless the user explicitly
  authorizes a persistence design.
- The agent does not control the OS. Any future action layer needs an allowlist,
  confirmation policy and audit log.

## Configuration and secrets

- `.env`, `config.yaml`, `*.log`, `.venv/`, `models/` and `runtime/` are local
  state and must never be committed.
- Never print, log, stage or paste API keys. Use `.env.example` with empty values.
- Public configuration changes go in `config.example.yaml`; keep the local
  `config.yaml` untouched unless the task explicitly requires tuning this
  machine.
- When adding a setting, update the `Settings` dataclass, loader defaults,
  `config.example.yaml`, the test fixture and relevant tests together.
- Screen captures and transcripts are potentially private. Do not add persistent
  capture/audio recording for debugging without explicit user approval.

## Windows and launchers

- Python 3.11 is the supported runtime.
- `start-agent.bat` is the setup-capable launcher and defaults to the GUI.
- `start-gui.bat` is the quiet post-setup launcher using `pythonw.exe`.
- Keep BAT files ASCII where practical and always CRLF. Validate any BAT change
  with real `cmd.exe`, not only text inspection:

  ```powershell
  cmd.exe /d /c "call start-agent.bat --help"
  ```

- CUDA DLL may live in `runtime/cuda12` or the system `PATH`. Do not commit the
  DLL files or model caches.

## Development workflow

- Begin audits read-only. Separate confirmed findings from hypotheses and
  proposed changes.
- Preserve unrelated user changes and local runtime state.
- Use `rg` / `rg --files` for repository search.
- Prefer narrow behavior-preserving patches. A large refactor needs a concrete
  runtime or maintainability benefit.
- Keep the package boundaries meaningful: configuration, local STT, cloud STT,
  Luna chat, Fish TTS, GUI, server and orchestration should not be collapsed into
  one module.
- Update `README.md` and `docs/ARCHITECTURE.md` whenever setup, data flow,
  fallback behavior or runtime boundaries change.
- Commit and push only when the user asks. The default branch is `main` and the
  remote is `origin`.

## Required verification

Run checks proportional to the change. Before a release or push, run all of:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m bandit -q -ll -r berangaria_agent
$pyFiles = rg --files -g '*.py'
.\.venv\Scripts\python.exe -m py_compile $pyFiles
cmd.exe /d /c "call start-agent.bat --help"
.\.venv\Scripts\python.exe -m build
```

The established baseline is 38 passing tests. If the count changes, explain why.
For audio or GPU changes, unit tests are necessary but not sufficient: perform a
short live smoke test and inspect `berangaria-agent.log` without committing it.

## Latency and quality claims

- Use `first_voice` as the primary responsiveness metric. `total` includes full
  playback and is not time-to-response.
- Report STT, screen, Luna and Fish separately. Do not call a change faster from
  one anecdotal turn.
- Prefer several live turns and summarize median plus range; use p95 only with a
  sample large enough to make it meaningful.
- Current local STT is already a small part of latency. Optimize Luna/Fish or
  pipeline overlap before sacrificing wake reliability for tens of milliseconds.

## Git hygiene

Before committing or pushing:

- inspect `git status` and the staged file list;
- run `git diff --cached --check`;
- confirm no credential-shaped values are staged;
- confirm runtime state remains ignored;
- ensure the local commit and `origin/main` agree after push;
- verify the Windows CI result when the workflow or runtime code changed.
