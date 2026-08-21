import asyncio
from dataclasses import replace

import pytest

from berangaria_agent.chat import ChatResult
from berangaria_agent.service import AgentService, InputError, TurnRequest
from berangaria_agent.transcription import TranscriptionError


class _Conversation:
    def __init__(self):
        self.calls = []
        self.reset_calls = 0

    async def reply(self, message, screen=None, screen_mime="image/jpeg"):
        self.calls.append((message, screen, screen_mime))
        return ChatResult("Ответ Бер", "Открыт редактор" if screen else "")

    def reset(self):
        self.reset_calls += 1


class _Transcriber:
    async def transcribe(self, audio, mime):
        return "Что на экране?"


class _Fish:
    audio_mime = "audio/wav"

    async def synthesize(self, text):
        return b"mp3"


def test_full_voice_screen_turn(settings):
    conversation = _Conversation()
    service = AgentService(
        settings,
        conversation=conversation,
        transcriber=_Transcriber(),
        fish=_Fish(),
    )
    result = asyncio.run(
        service.process(TurnRequest(audio=b"voice", screen=b"jpeg", screen_mime="image/jpeg"))
    )

    assert result.transcript == "Что на экране?"
    assert result.screen_description == "Открыт редактор"
    assert result.reply == "Ответ Бер"
    assert result.audio == b"mp3"
    assert result.audio_mime == "audio/wav"
    assert result.warnings == ()
    assert conversation.calls == [("Что на экране?", b"jpeg", "image/jpeg")]


def test_text_and_vision_survive_missing_fish(settings):
    text_settings = replace(settings, fish_api_key="", fish_voice_id="")
    conversation = _Conversation()
    service = AgentService(text_settings, conversation=conversation)

    result = asyncio.run(service.process(TurnRequest(text="Привет", screen=b"ignored")))

    assert result.reply == "Ответ Бер"
    assert result.screen_description == "Открыт редактор"
    assert result.audio is None
    assert result.warnings == ("Текстовый режим: Fish Audio не настроен",)
    assert conversation.calls == [("Привет", b"ignored", "image/jpeg")]


def test_background_can_skip_buffered_tts(settings):
    fish = _Fish()
    service = AgentService(settings, conversation=_Conversation(), fish=fish)

    result = asyncio.run(service.process(TurnRequest(text="Привет"), synthesize=False))

    assert result.reply == "Ответ Бер"
    assert result.audio is None
    assert result.audio_mime is None
    assert result.warnings == ()


def test_voice_transcription_failure_is_actionable(settings):
    class _BrokenTranscriber:
        async def transcribe(self, audio, mime):
            raise TranscriptionError("HTTP 503")

    service = AgentService(
        settings,
        conversation=_Conversation(),
        transcriber=_BrokenTranscriber(),
    )
    with pytest.raises(InputError, match="HTTP 503"):
        asyncio.run(service.process(TurnRequest(audio=b"voice")))


def test_empty_turn_is_rejected(settings):
    service = AgentService(settings, conversation=_Conversation())
    with pytest.raises(InputError, match="текст или запись"):
        asyncio.run(service.process(TurnRequest()))
