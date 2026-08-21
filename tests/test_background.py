import asyncio
import io
import wave

from berangaria_agent import background
from berangaria_agent.background import (
    WakePhraseMatcher,
    fit_screen_dimensions,
    needs_original_screen,
    pcm_to_wav,
    play_pcm_stream,
    play_wav,
)


def test_screen_dimensions_fit_without_upscaling():
    assert fit_screen_dimensions(2560, 1440, 1920, 1080) == (1920, 1080)
    assert fit_screen_dimensions(1920, 1200, 1920, 1080) == (1728, 1080)
    assert fit_screen_dimensions(1280, 720, 1920, 1080) == (1280, 720)


def test_text_reading_requests_keep_original_screen():
    assert needs_original_screen("Бер, прочитай, что написано на экране")
    assert needs_original_screen("Какая ошибка сейчас в терминале?")
    assert needs_original_screen("Read the error message")
    assert not needs_original_screen("Что сейчас на экране?")
    assert not needs_original_screen("Как у тебя дела?")


def test_wake_phrase_extracts_same_utterance_request():
    matcher = WakePhraseMatcher(
        ("бер", "берангария", "вер"),
        ("берт", "биар", "br"),
    )

    assert matcher.extract_request("Бер, что сейчас на экране?") == "что сейчас на экране"
    assert matcher.extract_request("Слушай, Берангария: открой контекст") == "открой контекст"
    assert matcher.extract_request("Берт, что за игра?") == "что за игра"
    assert matcher.extract_request("Биар") == ""
    assert matcher.extract_request("BR, ответь") == "ответь"
    assert matcher.extract_request("Это актёр Берт Ланкастер") is None
    assert matcher.extract_request("Обычный разговор") is None


def test_wake_phrase_can_arm_followup():
    matcher = WakePhraseMatcher(("бер",))
    assert matcher.extract_request("Бер!") == ""


def test_pcm_to_wav_builds_openrouter_friendly_mono_16khz():
    payload = pcm_to_wav(b"\x00\x00" * 480)

    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 480


def test_play_wav_streams_pcm_through_sounddevice(monkeypatch):
    payload = pcm_to_wav(b"\x01\x00" * 5000)
    writes: list[bytes] = []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            assert kwargs == {
                "samplerate": 16_000,
                "channels": 1,
                "dtype": "int16",
                "latency": "low",
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, chunk):
            writes.append(bytes(chunk))

    monkeypatch.setattr(background.sd, "RawOutputStream", FakeOutputStream)

    play_wav(payload)

    assert b"".join(writes) == b"\x01\x00" * 5000


def test_play_pcm_stream_handles_unaligned_network_chunks(monkeypatch):
    writes: list[bytes] = []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            assert kwargs["samplerate"] == 44_100
            assert kwargs["dtype"] == "int16"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, chunk):
            writes.append(bytes(chunk))

    async def chunks():
        yield b"\x01"
        yield b"\x00\x02"
        yield b"\x00"

    monkeypatch.setattr(background.sd, "RawOutputStream", FakeOutputStream)

    first_audio = asyncio.run(play_pcm_stream(chunks()))

    assert first_audio >= 0
    assert b"".join(writes) == b"\x01\x00\x02\x00"
