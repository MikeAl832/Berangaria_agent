import asyncio

from berangaria_agent import fish
from berangaria_agent.fish import FishClient, speech_text


def test_speech_text_strips_injected_controls_and_whitelists_emotion():
    assert speech_text("[angry] Привет   мир", emotion="calm", max_chars=100) == "[calm] Привет мир"
    assert speech_text("Привет", emotion="unknown", max_chars=100) == "Привет"


def test_speech_text_caps_length():
    assert speech_text("123456", emotion="", max_chars=4) == "1234"


def test_stream_pcm_posts_streaming_pcm_request(settings, monkeypatch):
    class Response:
        status_code = 200

        async def aiter_bytes(self):
            yield b"\x01\x00"
            yield b"\x02\x00"

    class StreamContext:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return False

    class Client:
        def __init__(self):
            self.request = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, method, url, *, headers, json):
            self.request = (method, url, headers, json)
            return StreamContext()

    client = Client()
    monkeypatch.setattr(fish.httpx, "AsyncClient", lambda **_kwargs: client)

    async def collect():
        return b"".join([chunk async for chunk in FishClient(settings).stream_pcm("Привет")])

    assert asyncio.run(collect()) == b"\x01\x00\x02\x00"
    method, url, _headers, payload = client.request
    assert method == "POST"
    assert url == fish.API_URL
    assert payload["format"] == "pcm"
    assert payload["sample_rate"] == fish.PCM_SAMPLE_RATE
