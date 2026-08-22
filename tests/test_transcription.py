import asyncio
import base64
from dataclasses import replace

import pytest

from berangaria_agent import transcription
from berangaria_agent.background import pcm_to_wav
from berangaria_agent.transcription import sanitize_transcript


class _Response:
    status_code = 200
    text = "ok"
    headers = {}

    def json(self):
        return {"text": "  Что на экране?  ", "usage": {"seconds": 1.2}}


class _Client:
    def __init__(self):
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None, headers=None, timeout=None):
        self.request = (url, json, headers)
        return _Response()


def test_transcribe_uses_openrouter_stt_with_existing_key(settings, monkeypatch):
    client = _Client()
    monkeypatch.setattr(transcription.httpx, "AsyncClient", lambda **kwargs: client)

    result = asyncio.run(
        transcription.TranscriptionClient(settings).transcribe(b"voice", "audio/webm")
    )

    assert result == "Что на экране?"
    url, payload, headers = client.request
    assert url == settings.openrouter_stt_url
    assert payload == {
        "model": settings.transcription_model,
        "input_audio": {
            "data": base64.b64encode(b"voice").decode("ascii"),
            "format": "webm",
        },
        "temperature": 0.0,
        "language": "ru",
        "provider": {
            "allow_fallbacks": True,
            "data_collection": "deny",
            "order": ["groq"],
        },
    }
    assert headers["Authorization"] == "Bearer openrouter-test"
    assert headers["X-OpenRouter-Title"] == settings.openrouter_title


def test_sanitize_transcript_removes_silence_hallucinations_and_noise_markers():
    raw = "*звук шума* Продолжение следует... Бер, что сейчас видно? Продолжение следует…"

    assert sanitize_transcript(raw) == "Бер, что сейчас видно?"


def test_sanitize_transcript_preserves_normal_speech():
    assert sanitize_transcript("Расскажи про музыку и шумоподавление") == (
        "Расскажи про музыку и шумоподавление"
    )


def test_wav_duration_limit_is_enforced_before_cloud_request(settings):
    limited = replace(settings, transcription_max_seconds=1)
    audio = pcm_to_wav(b"\x00\x00" * 16_000 * 2)

    with pytest.raises(transcription.TranscriptionError, match="длиннее допустимых"):
        asyncio.run(transcription.TranscriptionClient(limited).transcribe(audio, "audio/wav"))
