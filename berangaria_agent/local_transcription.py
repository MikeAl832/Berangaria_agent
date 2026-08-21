"""Local GPU Whisper transcription for the background voice loop."""

from __future__ import annotations

import io
import logging
import os
import shutil
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from berangaria_agent.config import Settings
from berangaria_agent.transcription import sanitize_transcript

logger = logging.getLogger(__name__)

_DLL_DIRECTORY_HANDLES: list[Any] = []


class LocalTranscriptionError(RuntimeError):
    """Local Whisper could not initialize or transcribe an utterance."""


class LocalNoSpeechError(LocalTranscriptionError):
    """Local Whisper found no usable speech in an utterance."""


@dataclass(frozen=True)
class LocalTranscript:
    text: str
    avg_logprob: float | None
    no_speech_probability: float | None

    def reliable_as_wake_only(self, settings: Settings) -> bool:
        if (
            self.no_speech_probability is not None
            and self.no_speech_probability
            > settings.local_transcription_wake_max_no_speech_prob
        ):
            return False
        return not (
            self.avg_logprob is not None
            and self.avg_logprob < settings.local_transcription_wake_min_avg_logprob
        )


def _runtime_path(settings: Settings) -> Path:
    configured = Path(settings.local_transcription_cuda_path)
    return configured if configured.is_absolute() else settings.project_root / configured


def configure_cuda_runtime(settings: Settings) -> Path | None:
    if settings.local_transcription_device != "cuda":
        return None
    runtime = _runtime_path(settings).resolve()
    required = (runtime / "cublas64_12.dll", runtime / "cudnn64_9.dll")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        if all(shutil.which(name) for name in ("cublas64_12.dll", "cudnn64_9.dll")):
            return None
        raise LocalTranscriptionError(
            f"Не найдены CUDA 12/cuDNN 9 библиотеки в {runtime} или PATH: "
            f"{', '.join(missing)}"
        )
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime)))
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(runtime).casefold() not in {part.casefold() for part in path_parts}:
        os.environ["PATH"] = str(runtime) + os.pathsep + os.environ.get("PATH", "")
    return runtime


def _silence_wav(duration_seconds: float = 0.35) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * round(16_000 * duration_seconds))
    return output.getvalue()


class LocalWhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.load_seconds = 0.0

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        started = time.monotonic()
        configure_cuda_runtime(self.settings)
        try:
            from faster_whisper import WhisperModel

            download_root = self.settings.project_root / "models" / "faster-whisper"
            download_root.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                self.settings.local_transcription_model,
                device=self.settings.local_transcription_device,
                compute_type=self.settings.local_transcription_compute_type,
                download_root=str(download_root),
            )
            self._transcribe_raw(_silence_wav())
        except Exception as exc:
            self._model = None
            raise LocalTranscriptionError(
                f"Не удалось запустить local Whisper: {exc}"
            ) from exc
        self.load_seconds = time.monotonic() - started
        logger.info(
            "Local Whisper готов: model=%s device=%s compute=%s load=%.3fs",
            self.settings.local_transcription_model,
            self.settings.local_transcription_device,
            self.settings.local_transcription_compute_type,
            self.load_seconds,
        )

    def _transcribe_raw(self, wav_audio: bytes) -> LocalTranscript:
        if self._model is None:
            raise LocalTranscriptionError("Local Whisper ещё не загружен")
        segments, _ = self._model.transcribe(
            io.BytesIO(wav_audio),
            language=self.settings.transcription_language or "ru",
            beam_size=self.settings.local_transcription_beam_size,
            best_of=self.settings.local_transcription_beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            hotwords=self.settings.local_transcription_hotwords or None,
        )
        usable_segments = [segment for segment in segments if segment.text.strip()]
        text = " ".join(segment.text.strip() for segment in usable_segments)
        weighted_logprob = 0.0
        logprob_weight = 0.0
        no_speech_probabilities: list[float] = []
        for segment in usable_segments:
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None:
                duration = max(
                    0.01,
                    float(getattr(segment, "end", 0.0))
                    - float(getattr(segment, "start", 0.0)),
                )
                weighted_logprob += float(avg_logprob) * duration
                logprob_weight += duration
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            if no_speech_prob is not None:
                no_speech_probabilities.append(float(no_speech_prob))
        return LocalTranscript(
            text=text,
            avg_logprob=(weighted_logprob / logprob_weight if logprob_weight else None),
            no_speech_probability=(
                max(no_speech_probabilities) if no_speech_probabilities else None
            ),
        )

    def transcribe_with_metadata(self, wav_audio: bytes) -> LocalTranscript:
        if not wav_audio.startswith(b"RIFF"):
            raise LocalTranscriptionError("Local Whisper принимает только WAV")
        try:
            with self._lock:
                raw_result = self._transcribe_raw(wav_audio)
                result = LocalTranscript(
                    text=sanitize_transcript(raw_result.text),
                    avg_logprob=raw_result.avg_logprob,
                    no_speech_probability=raw_result.no_speech_probability,
                )
        except LocalTranscriptionError:
            raise
        except Exception as exc:
            raise LocalTranscriptionError(f"Ошибка local Whisper: {exc}") from exc
        if not result.text:
            raise LocalNoSpeechError("В записи распознаны только тишина или шум")
        return result

    def transcribe(self, wav_audio: bytes) -> str:
        return self.transcribe_with_metadata(wav_audio).text
