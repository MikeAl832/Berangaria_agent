import asyncio
import io
import threading
import wave

import pytest

from berangaria_agent import background
from berangaria_agent.background import (
    BackgroundStopped,
    WakePhraseMatcher,
    _await_or_stop,
    _log_text,
    _stream_voice_response,
    fit_screen_dimensions,
    needs_original_screen,
    pcm_to_wav,
    play_pcm_stream,
)
from berangaria_agent.chat import ChatResult


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

    metrics = asyncio.run(play_pcm_stream(chunks()))

    assert metrics.first_chunk_seconds >= 0
    assert metrics.first_output_seconds >= metrics.first_chunk_seconds
    assert b"".join(writes) == b"\x01\x00\x02\x00"


def test_play_pcm_stream_enforces_first_chunk_deadline():
    async def slow_chunks():
        await asyncio.sleep(0.05)
        yield b"\x01\x00"

    with pytest.raises(RuntimeError, match="не начал поток"):
        asyncio.run(play_pcm_stream(slow_chunks(), first_chunk_timeout_seconds=0.01))


def test_model_text_and_fish_audio_are_pipelined(monkeypatch):
    writes = []

    class FakeOutputStream:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, chunk):
            writes.append(bytes(chunk))

    class Conversation:
        async def stream_reply(self, _message, _screen, _mime, *, on_delta):
            on_delta("Первая фраза. ")
            await asyncio.sleep(0.01)
            on_delta("Вторая.")
            return ChatResult("Первая фраза. Вторая.")

    class Fish:
        class Settings:
            fish_max_chars = 600

        settings = Settings()

        def __init__(self):
            self.segments = []

        async def stream_pcm(self, text):
            self.segments.append(text)
            yield b"\x01\x00"

    monkeypatch.setattr(background.sd, "RawOutputStream", FakeOutputStream)

    fish = Fish()
    result = asyncio.run(
        _stream_voice_response(
            Conversation(),
            fish,
            "Запрос",
            b"screen",
            model_timeout_seconds=1,
            fish_first_audio_timeout_seconds=1,
            status_callback=None,
        )
    )

    assert result.chat.reply == "Первая фраза. Вторая."
    assert result.model_first_text_seconds <= result.model_total_seconds
    assert result.model_first_text_seconds <= result.speech_ready_seconds
    assert fish.segments == ["Первая фраза.", "Вторая."]
    assert b"".join(writes) == b"\x01\x00\x01\x00"


def test_active_turn_can_be_cancelled_by_stop_event():
    stop = threading.Event()

    async def wait_forever():
        await asyncio.Event().wait()

    async def scenario():
        asyncio.get_running_loop().call_later(0.01, stop.set)
        await _await_or_stop(wait_forever(), stop)

    with pytest.raises(BackgroundStopped):
        asyncio.run(scenario())


def test_content_logging_is_redacted_by_default(settings, caplog):
    caplog.set_level("INFO", logger="berangaria_agent.background")

    _log_text(settings, "Распознано", "секретная фраза")

    assert "секретная фраза" not in caplog.text
    assert "chars=15" in caplog.text
