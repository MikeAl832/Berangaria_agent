"""Async Fish Audio client for browser-friendly MP3 replies."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

import httpx

from berangaria_agent.config import Settings

API_URL = "https://api.fish.audio/v1/tts"
PCM_SAMPLE_RATE = 44_100
_CONTROL_BRACKETS = re.compile(r"\[[^\]\n]{1,80}\]")
_ALLOWED_EMOTIONS = {
    "calm",
    "sarcastic",
    "disdainful",
    "bored",
    "indifferent",
    "confident",
    "sighing",
    "chuckling",
}
_AUDIO_MIMES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
}


class FishError(RuntimeError):
    """Fish Audio did not synthesize a reply."""


def speech_text(text: object, *, emotion: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _CONTROL_BRACKETS.sub(" ", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    normalized_emotion = emotion.strip().lower()
    if cleaned and normalized_emotion in _ALLOWED_EMOTIONS:
        return f"[{normalized_emotion}] {cleaned}"
    return cleaned


class FishClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def synthesize(self, text: str) -> bytes:
        if not self.settings.fish_ready:
            raise FishError("FISH_API_KEY или FISH_VOICE_ID не задан")
        prepared = speech_text(
            text,
            emotion=self.settings.fish_emotion,
            max_chars=self.settings.fish_max_chars,
        )
        if not prepared:
            raise FishError("Пустой текст для озвучивания")
        headers = {
            "Authorization": f"Bearer {self.settings.fish_api_key}",
            "Content-Type": "application/json",
            "model": self.settings.fish_model,
        }
        payload = {
            "text": prepared,
            "reference_id": self.settings.fish_voice_id,
            "format": self.settings.fish_format,
            "latency": "low",
            "normalize": True,
        }
        last_error = "неизвестная ошибка"
        async with httpx.AsyncClient(timeout=self.settings.fish_timeout_seconds) as client:
            for attempt in range(2):
                try:
                    response = await client.post(API_URL, headers=headers, json=payload)
                except httpx.RequestError as exc:
                    last_error = str(exc) or exc.__class__.__name__
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    break
                if response.status_code == 200:
                    if not response.content:
                        raise FishError("Fish Audio вернул пустой файл")
                    return response.content
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in {408, 425, 429, 500, 502, 503, 504} or attempt:
                    break
                await asyncio.sleep(0.5)
        raise FishError(f"Fish Audio недоступен: {last_error}")

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw mono PCM16 as Fish generates it, without buffering a full WAV."""
        if not self.settings.fish_ready:
            raise FishError("FISH_API_KEY или FISH_VOICE_ID не задан")
        prepared = speech_text(
            text,
            emotion=self.settings.fish_emotion,
            max_chars=self.settings.fish_max_chars,
        )
        if not prepared:
            raise FishError("Пустой текст для озвучивания")
        headers = {
            "Authorization": f"Bearer {self.settings.fish_api_key}",
            "Content-Type": "application/json",
            "model": self.settings.fish_model,
        }
        payload = {
            "text": prepared,
            "reference_id": self.settings.fish_voice_id,
            "format": "pcm",
            "sample_rate": PCM_SAMPLE_RATE,
            "latency": "low",
            "normalize": True,
        }
        last_error = "неизвестная ошибка"
        stream_timeout = httpx.Timeout(
            self.settings.fish_timeout_seconds,
            read=min(4.0, self.settings.fish_timeout_seconds),
        )
        async with httpx.AsyncClient(timeout=stream_timeout) as client:
            for attempt in range(2):
                received_audio = False
                try:
                    async with client.stream(
                        "POST", API_URL, headers=headers, json=payload
                    ) as response:
                        if response.status_code == 200:
                            async for chunk in response.aiter_bytes():
                                if chunk:
                                    received_audio = True
                                    yield chunk
                            if not received_audio:
                                raise FishError("Fish Audio вернул пустой PCM-поток")
                            return
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        last_error = f"HTTP {response.status_code}: {body[:200]}"
                        retryable = response.status_code in {408, 425, 429, 500, 502, 503, 504}
                        if not retryable or attempt:
                            break
                except httpx.RequestError as exc:
                    last_error = str(exc) or exc.__class__.__name__
                    if received_audio or attempt:
                        break
                await asyncio.sleep(0.5)
        raise FishError(f"Fish Audio недоступен: {last_error}")

    @property
    def audio_mime(self) -> str:
        return _AUDIO_MIMES.get(self.settings.fish_format, "application/octet-stream")
