"""Headless Windows voice loop: VAD, wake phrase, screenshot, reply and playback."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import tempfile
import threading
import time
import urllib.request
import wave
import winsound
import zipfile
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

import mss
import mss.tools
import sounddevice as sd
import vosk
import webrtcvad
from PIL import Image, ImageFilter

from berangaria_agent.config import Settings
from berangaria_agent.fish import PCM_SAMPLE_RATE, FishError
from berangaria_agent.local_transcription import (
    LocalNoSpeechError,
    LocalTranscript,
    LocalTranscriptionError,
    LocalWhisperTranscriber,
)
from berangaria_agent.service import AgentService, InputError, TurnRequest
from berangaria_agent.transcription import TranscriptionClient, TranscriptionError

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str, str], None]

_SAMPLE_RATE = 16_000
_CHANNELS = 1
_SAMPLE_WIDTH = 2
_FRAME_MS = 30
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_MS // 1000
_WAKE_MODEL_NAME = "vosk-model-small-ru-0.22"
_WAKE_MODEL_URL = f"https://alphacephei.com/vosk/models/{_WAKE_MODEL_NAME}.zip"


@dataclass(frozen=True)
class RecordedUtterance:
    wav: bytes
    duration_seconds: float
    trailing_silence_seconds: float


def configure_background_logging(root: Path) -> Path:
    log_path = root / "berangaria-agent.log"
    package_logger = logging.getLogger("berangaria_agent")
    package_logger.setLevel(logging.INFO)
    if not any(isinstance(handler, RotatingFileHandler) for handler in package_logger.handlers):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        package_logger.addHandler(handler)
    return log_path


def input_devices() -> list[tuple[int, str]]:
    return [
        (index, str(device["name"]))
        for index, device in enumerate(sd.query_devices())
        if int(device["max_input_channels"]) > 0
    ]


def resolve_microphone(query: str) -> tuple[int, str]:
    devices = input_devices()
    if not devices:
        raise RuntimeError("Windows не видит ни одного устройства записи")
    normalized = query.strip().casefold()
    if normalized:
        exact = [device for device in devices if device[1].casefold() == normalized]
        partial = [device for device in devices if normalized in device[1].casefold()]
        matches = exact or partial
        if matches:
            return matches[0]
        raise RuntimeError(f"Микрофон '{query}' не найден. Запусти --list-audio-devices")
    default_input = int(sd.default.device[0])
    for device in devices:
        if device[0] == default_input:
            return device
    return devices[0]


def pcm_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(_CHANNELS)
        wav.setsampwidth(_SAMPLE_WIDTH)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def resolve_wake_model_path(settings: Settings) -> Path:
    configured = Path(settings.wake_model_path)
    return configured if configured.is_absolute() else settings.project_root / configured


def install_wake_model(settings: Settings) -> Path:
    destination = resolve_wake_model_path(settings)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю локальную wake-word модель ({_WAKE_MODEL_NAME}, около 45 МБ)...")
    with tempfile.TemporaryDirectory(prefix=".wake-model-", dir=destination.parent) as temp:
        temp_root = Path(temp)
        archive_path = temp_root / "model.zip"
        # The URL is a fixed official HTTPS asset, never user-controlled.
        urllib.request.urlretrieve(  # nosec B310
            _WAKE_MODEL_URL,
            archive_path,
        )
        extract_root = temp_root / "extracted"
        extract_root.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (extract_root / member.filename).resolve()
                if not target.is_relative_to(extract_root.resolve()):
                    raise RuntimeError("Архив wake-word модели содержит небезопасный путь")
            archive.extractall(extract_root)
        extracted = extract_root / _WAKE_MODEL_NAME
        if not extracted.is_dir():
            raise RuntimeError("В архиве не найдена ожидаемая папка wake-word модели")
        extracted.rename(destination)
    return destination


class VoiceActivityRecorder:
    def __init__(
        self,
        device: int,
        *,
        aggressiveness: int,
        silence_ms: int,
        max_seconds: int,
    ) -> None:
        self.device = device
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames = max(1, silence_ms // _FRAME_MS)
        self.max_frames = max(1, max_seconds * 1000 // _FRAME_MS)

    def listen(
        self,
        max_wait_seconds: float | None = None,
        *,
        stop_event: threading.Event | None = None,
    ) -> RecordedUtterance | None:
        started_at = time.monotonic()
        pre_roll: deque[bytes] = deque(maxlen=10)
        voice_window: deque[bool] = deque(maxlen=5)
        frames: list[bytes] = []
        triggered = False
        trailing_silence = 0

        with sd.RawInputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_FRAME_SAMPLES,
            device=self.device,
            channels=_CHANNELS,
            dtype="int16",
            latency="low",
        ) as stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                data, overflowed = stream.read(_FRAME_SAMPLES)
                if overflowed:
                    logger.warning("Переполнение входного аудиобуфера")
                frame = bytes(data)
                speech = self.vad.is_speech(frame, _SAMPLE_RATE)

                if not triggered:
                    if (
                        max_wait_seconds is not None
                        and time.monotonic() - started_at >= max_wait_seconds
                    ):
                        return None
                    pre_roll.append(frame)
                    voice_window.append(speech)
                    if len(voice_window) == voice_window.maxlen and sum(voice_window) >= 3:
                        triggered = True
                        frames.extend(pre_roll)
                    continue

                frames.append(frame)
                trailing_silence = 0 if speech else trailing_silence + 1
                if trailing_silence >= self.silence_frames or len(frames) >= self.max_frames:
                    break

        return RecordedUtterance(
            wav=pcm_to_wav(b"".join(frames)),
            duration_seconds=len(frames) * _FRAME_MS / 1000,
            trailing_silence_seconds=trailing_silence * _FRAME_MS / 1000,
        )


class WakePhraseMatcher:
    def __init__(
        self,
        phrases: tuple[str, ...],
        aliases: tuple[str, ...] = (),
    ) -> None:
        normalized = [self._normalize(phrase) for phrase in phrases]
        self.phrases = tuple(
            sorted((phrase for phrase in normalized if phrase), key=len, reverse=True)
        )
        self.aliases = frozenset(
            alias for phrase in aliases if (alias := self._normalize(phrase))
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.casefold().replace("ё", "е")
        return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", text).split())

    def extract_request(self, transcript: str) -> str | None:
        normalized = self._normalize(transcript)
        for phrase in self.phrases:
            match = re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", normalized)
            if match:
                return normalized[match.end() :].strip()
        first_word, separator, remainder = normalized.partition(" ")
        if first_word in self.aliases:
            return remainder.strip() if separator else ""
        return None


class LocalWakeDetector:
    def __init__(self, model_path: Path, phrases: tuple[str, ...]) -> None:
        if not model_path.is_dir():
            raise RuntimeError(
                f"Локальная wake-word модель не найдена: {model_path}. Запусти --install-wake-model"
            )
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_path))
        self.matcher = WakePhraseMatcher(phrases)

    def detected(self, wav_audio: bytes) -> bool:
        with wave.open(io.BytesIO(wav_audio), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError("Wake-word принимает только mono PCM16")
            recognizer = vosk.KaldiRecognizer(self.model, wav.getframerate())
            while chunk := wav.readframes(4000):
                recognizer.AcceptWaveform(chunk)
        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "") if isinstance(result, dict) else ""
        logger.debug("Vosk fallback распознал: %s", text)
        return isinstance(text, str) and self.matcher.extract_request(text) is not None


def prepare_wake_detector(settings: Settings) -> LocalWakeDetector:
    """Load the cloud-STT fallback detector, downloading its small model if needed."""
    model_path = resolve_wake_model_path(settings)
    if not model_path.is_dir():
        model_path = install_wake_model(settings)
    return LocalWakeDetector(model_path, settings.wake_phrases)


_ORIGINAL_SCREEN_PATTERNS = (
    r"\bпрочита(?:й|йте)\b",
    r"\b(?:что|чего)\s+(?:там\s+)?написан",
    r"\b(?:ошибк|надпис|сообщени|текст|код)\w*\s+(?:на\s+)?экра",
    r"\b(?:ошибк|надпис|сообщени|текст)\w*\b",
    r"\b(?:терминал|консол)\w*\b",
    r"\b(?:мелк|увелич)\w*\b",
    r"\bread\s+(?:the\s+)?(?:screen|text|error)\b",
    r"\bwhat\s+(?:does|is)\s+(?:it|the\s+screen)\s+(?:say|showing)\b",
    r"\b(?:error\s+message|small\s+text|terminal|console)\b",
)


def needs_original_screen(request_text: str) -> bool:
    normalized = request_text.casefold()
    return any(re.search(pattern, normalized) for pattern in _ORIGINAL_SCREEN_PATTERNS)


def fit_screen_dimensions(
    width: int,
    height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize_screen_png(
    rgb: bytes,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> bytes:
    image = Image.frombytes("RGB", source_size, rgb)
    resized = image.resize(target_size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    sharpened = resized.filter(ImageFilter.UnsharpMask(radius=0.6, percent=35, threshold=2))
    output = io.BytesIO()
    sharpened.save(output, format="PNG", compress_level=4)
    return output.getvalue()


def capture_screen_png(
    monitor_index: int,
    max_width: int | None = None,
    max_height: int | None = None,
) -> bytes:
    with mss.mss() as capture:
        if monitor_index >= len(capture.monitors):
            raise RuntimeError(
                f"Монитор {monitor_index} не найден; доступно {len(capture.monitors) - 1}"
            )
        screenshot = capture.grab(capture.monitors[monitor_index])
        if max_width is not None and max_height is not None:
            target_size = fit_screen_dimensions(
                screenshot.width,
                screenshot.height,
                max_width,
                max_height,
            )
            if target_size != screenshot.size:
                return _resize_screen_png(screenshot.rgb, screenshot.size, target_size)
        return mss.tools.to_png(screenshot.rgb, screenshot.size)


def play_wav(audio: bytes) -> None:
    if not audio.startswith(b"RIFF"):
        raise RuntimeError("Fish Audio вернул не WAV; проверь fish_format: wav")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            _play_pcm_wav(wav)
    except (EOFError, wave.Error) as exc:
        raise RuntimeError(f"Fish Audio вернул повреждённый WAV: {exc}") from exc


def _play_pcm_wav(wav: wave.Wave_read) -> None:
    if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
        raise RuntimeError("Для воспроизведения нужен несжатый PCM16 WAV")
    channels = wav.getnchannels()
    sample_rate = wav.getframerate()
    if channels not in {1, 2} or sample_rate <= 0:
        raise RuntimeError("Fish Audio вернул неподдерживаемые параметры WAV")
    try:
        with sd.RawOutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            latency="low",
        ) as output:
            while chunk := wav.readframes(4096):
                output.write(chunk)
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Windows не смог воспроизвести голос: {exc}") from exc


async def play_pcm_stream(
    chunks: AsyncIterator[bytes],
    *,
    on_first_audio: Callable[[float], None] | None = None,
) -> float:
    """Play a Fish PCM stream and return seconds until its first audio bytes."""
    started = time.monotonic()
    iterator = chunks.__aiter__()
    try:
        first_chunk = await anext(iterator)
    except StopAsyncIteration as exc:
        raise RuntimeError("Fish Audio вернул пустой PCM-поток") from exc
    first_audio_seconds = time.monotonic() - started
    if on_first_audio is not None:
        on_first_audio(first_audio_seconds)

    pending = b""
    try:
        with sd.RawOutputStream(
            samplerate=PCM_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            latency="low",
        ) as output:
            for chunk in (first_chunk,):
                pending += chunk
                frame_bytes = len(pending) - len(pending) % 2
                if frame_bytes:
                    output.write(pending[:frame_bytes])
                    pending = pending[frame_bytes:]
            async for chunk in iterator:
                pending += chunk
                frame_bytes = len(pending) - len(pending) % 2
                if frame_bytes:
                    output.write(pending[:frame_bytes])
                    pending = pending[frame_bytes:]
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Windows не смог воспроизвести голос: {exc}") from exc
    if pending:
        raise RuntimeError("Fish Audio оборвал PCM-поток посреди аудиосэмпла")
    return first_audio_seconds


def _percentile(samples: deque[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _publish(callback: StatusCallback | None, state: str, detail: str = "") -> None:
    if callback is not None:
        callback(state, detail)


async def run_background(
    settings: Settings,
    *,
    stop_event: threading.Event | None = None,
    status_callback: StatusCallback | None = None,
) -> None:
    if not settings.fish_ready:
        raise ValueError("Для фонового голосового режима нужны FISH_API_KEY и FISH_VOICE_ID")
    if settings.fish_format != "wav":
        raise ValueError("Для фонового режима установи fish_format: wav")

    log_path = configure_background_logging(settings.project_root)
    device_id, device_name = resolve_microphone(settings.microphone_device)
    recorder = VoiceActivityRecorder(
        device_id,
        aggressiveness=settings.vad_aggressiveness,
        silence_ms=settings.vad_silence_ms,
        max_seconds=settings.vad_max_seconds,
    )
    matcher = WakePhraseMatcher(settings.wake_phrases, settings.wake_aliases)
    wake_detector: LocalWakeDetector | None = None
    remote_transcriber = TranscriptionClient(settings)
    local_transcriber: LocalWhisperTranscriber | None = None
    if settings.transcription_backend == "local":
        _publish(status_callback, "transcribing", "Загружаю local Whisper на GPU…")
        candidate = LocalWhisperTranscriber(settings)
        try:
            await asyncio.to_thread(candidate.load)
            local_transcriber = candidate
        except LocalTranscriptionError as exc:
            logger.exception("Local Whisper не запустился")
            if not settings.local_transcription_fallback_to_openrouter:
                raise RuntimeError(str(exc)) from exc
            _publish(status_callback, "warning", f"Local Whisper: {exc}; использую OpenRouter")
    if local_transcriber is None:
        _publish(status_callback, "activating", "Подготавливаю резервную модель вызова…")
        wake_detector = await asyncio.to_thread(prepare_wake_detector, settings)
    service = AgentService(settings)
    armed_until = 0.0
    stop = stop_event or threading.Event()
    first_voice_samples: deque[float] = deque(maxlen=50)

    print(f"Berangaria слушает: {device_name}")
    print(f"Фразы вызова: {', '.join(settings.wake_phrases)}")
    print(f"Лог: {log_path}")
    logger.info(
        "Фоновый агент запущен; microphone=%s monitor=%s", device_name, settings.screen_monitor
    )
    _publish(status_callback, "listening", device_name)

    while not stop.is_set():
        followup_timeout = max(0.0, armed_until - time.monotonic()) if armed_until else None
        if armed_until and followup_timeout == 0:
            armed_until = 0.0
            continue
        _publish(
            status_callback,
            "listening",
            "Жду следующую фразу" if armed_until else f"Микрофон: {device_name}",
        )
        utterance = await asyncio.to_thread(
            recorder.listen,
            followup_timeout,
            stop_event=stop,
        )
        if stop.is_set():
            break
        if utterance is None:
            armed_until = 0.0
            continue

        recording_finished = time.monotonic()
        speech_ended_at = recording_finished - utterance.trailing_silence_seconds
        turn_started = speech_ended_at
        vad_seconds = utterance.trailing_silence_seconds
        wake_seconds = 0.0
        transcript = ""
        local_result: LocalTranscript | None = None
        if local_transcriber is not None:
            _publish(status_callback, "transcribing", "Распознаю локально…")
            stt_started = time.monotonic()
            try:
                local_result = await asyncio.to_thread(
                    local_transcriber.transcribe_with_metadata, utterance.wav
                )
                transcript = local_result.text
            except LocalNoSpeechError as exc:
                logger.info("Local Whisper пропустил шум: %s", exc)
                continue
            except LocalTranscriptionError as exc:
                logger.exception("Local Whisper не распознал фразу")
                if not settings.local_transcription_fallback_to_openrouter:
                    _publish(status_callback, "warning", f"Local STT: {exc}")
                    continue
                local_transcriber = None
                _publish(
                    status_callback,
                    "warning",
                    f"Local STT: {exc}; переключаюсь на OpenRouter",
                )
            stt_seconds = time.monotonic() - stt_started
            if transcript:
                wake_started = time.monotonic()
                extracted_request = matcher.extract_request(transcript)
                wake_only = extracted_request == ""
                locally_activated = bool(armed_until) or extracted_request is not None
                wake_seconds = time.monotonic() - wake_started
                logger.info(
                    "Local STT: audio=%.3fs avg_logprob=%s no_speech=%s wake_only=%s",
                    utterance.duration_seconds,
                    (
                        f"{local_result.avg_logprob:.3f}"
                        if local_result and local_result.avg_logprob is not None
                        else "n/a"
                    ),
                    (
                        f"{local_result.no_speech_probability:.3f}"
                        if local_result and local_result.no_speech_probability is not None
                        else "n/a"
                    ),
                    wake_only,
                )
                if (
                    not armed_until
                    and wake_only
                    and local_result is not None
                    and not local_result.reliable_as_wake_only(settings)
                ):
                    logger.info("Отброшено неуверенное одиночное wake-word: %s", transcript)
                    continue
                if not locally_activated:
                    logger.info("Пропущено без wake-word: %s", transcript)
                    continue

        if not transcript:
            if wake_detector is None:
                _publish(
                    status_callback,
                    "activating",
                    "Подготавливаю резервную модель вызова…",
                )
                wake_detector = await asyncio.to_thread(prepare_wake_detector, settings)
            _publish(status_callback, "activating", "Проверяю локальное слово вызова…")
            wake_started = time.monotonic()
            locally_activated = bool(armed_until) or await asyncio.to_thread(
                wake_detector.detected, utterance.wav
            )
            wake_seconds = time.monotonic() - wake_started
            if not locally_activated:
                continue
            _publish(status_callback, "transcribing", "Распознаю через OpenRouter…")
            stt_started = time.monotonic()
            try:
                transcript = await remote_transcriber.transcribe(utterance.wav, "audio/wav")
            except TranscriptionError as exc:
                logger.warning("STT не распознал фразу: %s", exc)
                _publish(status_callback, "warning", f"STT: {exc}")
                continue
            stt_seconds = time.monotonic() - stt_started

        logger.info("Распознано: %s", transcript)
        _publish(status_callback, "heard", transcript)

        if armed_until:
            request_text = transcript.strip()
        else:
            request_text = matcher.extract_request(transcript)
            if request_text is None:
                request_text = transcript.strip()
        armed_until = 0.0

        if not request_text:
            await asyncio.to_thread(winsound.MessageBeep, winsound.MB_OK)
            armed_until = time.monotonic() + settings.wake_followup_seconds
            continue

        _publish(status_callback, "thinking", request_text)
        try:
            screen_started = time.monotonic()
            original_screen = (
                settings.screen_original_for_text_requests
                and needs_original_screen(request_text)
            )
            screen = await asyncio.to_thread(
                capture_screen_png,
                settings.screen_monitor,
                None if original_screen else settings.screen_max_width,
                None if original_screen else settings.screen_max_height,
            )
            screen_seconds = time.monotonic() - screen_started
            logger.info(
                "Снимок экрана: mode=%s bytes=%d",
                "original" if original_screen else "resized",
                len(screen),
            )
            luna_started = time.monotonic()
            result = await service.process(
                TurnRequest(text=request_text, screen=screen, screen_mime="image/png"),
                synthesize=False,
            )
            luna_seconds = time.monotonic() - luna_started
        except (InputError, RuntimeError) as exc:
            logger.exception("Не удалось обработать голосовой ход")
            print(f"Ошибка: {exc}")
            _publish(status_callback, "error", str(exc))
            continue

        logger.info("Ответ: %s", result.reply)
        _publish(status_callback, "reply", result.reply)
        for warning in result.warnings:
            logger.warning("Ход завершён с предупреждением: %s", warning)
            _publish(status_callback, "warning", warning)
        _publish(status_callback, "speaking", "Fish готовит первые аудиоданные…")
        tts_started = time.monotonic()
        try:
            first_audio_seconds = await play_pcm_stream(
                service.fish.stream_pcm(result.reply),
                on_first_audio=lambda delay: _publish(
                    status_callback,
                    "speaking",
                    f"Говорю · Fish начал поток за {delay:.1f} с",
                ),
            )
        except (FishError, RuntimeError) as exc:
            logger.exception("Не удалось воспроизвести потоковый TTS")
            _publish(status_callback, "error", str(exc))
            continue
        tts_total_seconds = time.monotonic() - tts_started
        first_voice_seconds = tts_started + first_audio_seconds - turn_started
        total_seconds = time.monotonic() - turn_started
        first_voice_samples.append(first_voice_seconds)
        metric = (
            f"После конца речи {first_voice_seconds:.1f} с · VAD {vad_seconds:.1f} · "
            f"wake {wake_seconds:.1f} · STT {stt_seconds:.1f} · "
            f"Luna {luna_seconds:.1f} · Fish {first_audio_seconds:.1f}"
        )
        if len(first_voice_samples) >= 5:
            metric += (
                f" · p50 {_percentile(first_voice_samples, 0.5):.1f}"
                f" / p95 {_percentile(first_voice_samples, 0.95):.1f}"
            )
        _publish(status_callback, "metrics", metric)
        logger.info(
            "Задержка хода: vad=%.3fs wake=%.3fs stt=%.3fs screen=%.3fs luna=%.3fs "
            "fish_first=%.3fs first_voice=%.3fs tts_playback=%.3fs total=%.3fs",
            vad_seconds,
            wake_seconds,
            stt_seconds,
            screen_seconds,
            luna_seconds,
            first_audio_seconds,
            first_voice_seconds,
            tts_total_seconds,
            total_seconds,
        )

    logger.info("Фоновый агент остановлен")
    _publish(status_callback, "stopped", "Агент остановлен")
