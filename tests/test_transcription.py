import asyncio
import base64

from berangaria_agent import transcription
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

    async def post(self, url, json=None, headers=None):
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
            "order": ["groq"],
            "allow_fallbacks": True,
        },
    }
    assert headers["Authorization"] == "Bearer openrouter-test"


def test_sanitize_transcript_removes_silence_hallucinations_and_noise_markers():
    raw = "*звук шума* Продолжение следует... Бер, что сейчас видно? Продолжение следует…"

    assert sanitize_transcript(raw) == "Бер, что сейчас видно?"


def test_sanitize_transcript_preserves_normal_speech():
    assert sanitize_transcript("Расскажи про музыку и шумоподавление") == (
        "Расскажи про музыку и шумоподавление"
    )
