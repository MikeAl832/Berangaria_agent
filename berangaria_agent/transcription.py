"""Speech-to-text through OpenRouter using the existing API key."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
from collections.abc import Mapping

import httpx

from berangaria_agent.config import Settings

logger = logging.getLogger(__name__)

_AUDIO_FORMATS = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
_HALLUCINATION_PHRASES = re.compile(
    r"(?i)(?:продолжение\s+следует|спасибо\s+за\s+просмотр|"
    r"субтитры\s+(?:сделал|создал|подготовил)[^.?!…]*)[\s.,!?…]*"
)
_NON_SPEECH_MARKERS = re.compile(
    r"(?ix)(?:\*|\[|\()\s*"
    r"(?:звук\s+)?(?:шум[а-я]*|музык[а-я]*|тишин[а-я]*|дыхани[а-я]*|"
    r"каш[а-я]*|смех[а-я]*|аплодисмент[а-я]*|noise|music|silence)"
    r"\s*(?:\*|\]|\))"
)


class TranscriptionError(RuntimeError):
    """OpenRouter speech-to-text failed or returned unusable content."""


def transcription_text(data: object) -> str:
    if not isinstance(data, Mapping):
        return ""
    value = data.get("text")
    return value.strip() if isinstance(value, str) else ""


def sanitize_transcript(text: str) -> str:
    """Remove common Whisper silence hallucinations, not actual speech content."""
    cleaned = _NON_SPEECH_MARKERS.sub(" ", text)
    cleaned = _HALLUCINATION_PHRASES.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?…])", r"\1", cleaned)
    cleaned = cleaned.strip(" \t\r\n—-*")
    return cleaned if re.search(r"[a-zа-яё0-9]", cleaned, re.IGNORECASE) else ""


class TranscriptionClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _normalized_audio(self, audio: bytes, audio_format: str) -> tuple[bytes, str]:
        if not self.settings.transcription_normalize_audio or audio_format == "wav":
            return audio, audio_format
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning(
                "ffmpeg не найден; отправляю %s в OpenRouter STT без нормализации",
                audio_format,
            )
            return audio, audio_format
        process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-t",
            str(self.settings.transcription_max_seconds),
            "-f",
            "wav",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        normalized, stderr = await process.communicate(audio)
        if process.returncode or not normalized.startswith(b"RIFF"):
            details = stderr.decode("utf-8", errors="replace").strip()[:200]
            raise TranscriptionError(
                f"Не удалось декодировать запись {audio_format} через ffmpeg: "
                f"{details or f'код {process.returncode}'}"
            )
        return normalized, "wav"

    async def transcribe(self, audio: bytes, mime: str) -> str:
        audio_format = _AUDIO_FORMATS.get(mime)
        if audio_format is None:
            raise TranscriptionError(f"Неподдерживаемый формат аудио: {mime}")
        audio, audio_format = await self._normalized_audio(audio, audio_format)
        payload = {
            "model": self.settings.transcription_model,
            "input_audio": {
                "data": base64.b64encode(audio).decode("ascii"),
                "format": audio_format,
            },
            "temperature": self.settings.transcription_temperature,
        }
        if self.settings.transcription_language:
            payload["language"] = self.settings.transcription_language
        if self.settings.transcription_provider != "auto":
            payload["provider"] = {
                "order": [self.settings.transcription_provider],
                "allow_fallbacks": self.settings.transcription_provider_allow_fallbacks,
            }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_referer:
            headers["HTTP-Referer"] = self.settings.openrouter_referer
        if self.settings.openrouter_title:
            headers["X-Title"] = self.settings.openrouter_title
        last_error = "неизвестная ошибка"
        async with httpx.AsyncClient(timeout=self.settings.transcription_timeout_seconds) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        self.settings.openrouter_stt_url,
                        json=payload,
                        headers=headers,
                    )
                except httpx.RequestError as exc:
                    last_error = str(exc) or exc.__class__.__name__
                    if attempt < 2:
                        await asyncio.sleep(min(2.0**attempt, 4.0))
                        continue
                    break
                if response.status_code == 200:
                    try:
                        text = transcription_text(response.json())
                    except ValueError as exc:
                        raise TranscriptionError("OpenRouter STT вернул невалидный JSON") from exc
                    text = sanitize_transcript(text)
                    if not text:
                        raise TranscriptionError("В записи распознаны только тишина или шум")
                    return text
                generation_id = response.headers.get("X-Generation-Id", "нет")
                last_error = (
                    f"HTTP {response.status_code}, generation={generation_id}, "
                    f"format={audio_format}, bytes={len(audio)}: {response.text[:200]}"
                )
                if response.status_code not in _RETRYABLE_STATUSES or attempt >= 2:
                    break
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(max(float(retry_after), 0.25), 10.0)
                except (TypeError, ValueError):
                    delay = min(2.0**attempt, 4.0)
                await asyncio.sleep(delay)
        raise TranscriptionError(f"OpenRouter STT недоступен: {last_error}")
