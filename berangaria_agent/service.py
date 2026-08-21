"""One voice/text turn backed by an optional current screenshot."""

from __future__ import annotations

from dataclasses import dataclass

from berangaria_agent.chat import Conversation
from berangaria_agent.config import Settings
from berangaria_agent.fish import FishClient, FishError
from berangaria_agent.transcription import TranscriptionClient, TranscriptionError


class InputError(ValueError):
    """The local browser sent an unusable turn."""


@dataclass(frozen=True)
class TurnRequest:
    text: str = ""
    audio: bytes | None = None
    audio_mime: str = "audio/webm"
    screen: bytes | None = None
    screen_mime: str = "image/jpeg"


@dataclass(frozen=True)
class TurnResult:
    transcript: str
    screen_description: str
    reply: str
    audio: bytes | None
    audio_mime: str | None
    warnings: tuple[str, ...] = ()


class AgentService:
    def __init__(
        self,
        settings: Settings,
        *,
        conversation: Conversation | None = None,
        transcriber: TranscriptionClient | None = None,
        fish: FishClient | None = None,
    ) -> None:
        self.settings = settings
        self.conversation = conversation or Conversation(settings)
        self.transcriber = transcriber or TranscriptionClient(settings)
        self.fish = fish or FishClient(settings)

    def reset(self) -> None:
        self.conversation.reset()

    async def process(self, request: TurnRequest, *, synthesize: bool = True) -> TurnResult:
        typed_text = request.text.strip()
        if not typed_text and not request.audio:
            raise InputError("Нужен текст или запись с микрофона")

        warnings: list[str] = []
        transcript = ""
        if request.audio:
            try:
                transcript = (
                    await self.transcriber.transcribe(request.audio, request.audio_mime)
                ).strip()
            except TranscriptionError as exc:
                raise InputError(f"Не удалось распознать речь: {exc}") from exc
            if not transcript and not typed_text:
                raise InputError("В записи не распознана речь")

        owner_message = typed_text
        if transcript:
            owner_message = (
                f"{typed_text}\n\nРаспознано с микрофона: {transcript}"
                if typed_text
                else transcript
            )
        chat_result = await self.conversation.reply(
            owner_message,
            request.screen,
            request.screen_mime,
        )
        if request.screen and not chat_result.screen_description:
            warnings.append("Luna не вернула описание снимка")

        speech: bytes | None = None
        speech_mime: str | None = None
        if synthesize and self.settings.fish_ready:
            try:
                speech = await self.fish.synthesize(chat_result.reply)
                speech_mime = self.fish.audio_mime
            except FishError as exc:
                warnings.append(f"Ответ не озвучен: {exc}")
        elif synthesize:
            warnings.append("Текстовый режим: Fish Audio не настроен")

        return TurnResult(
            transcript=transcript or typed_text,
            screen_description=chat_result.screen_description,
            reply=chat_result.reply,
            audio=speech,
            audio_mime=speech_mime,
            warnings=tuple(warnings),
        )
