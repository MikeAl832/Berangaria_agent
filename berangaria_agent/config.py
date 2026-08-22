"""Standalone configuration with no dependency on the Telegram bot."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _as_int(value: object, default: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _as_float(value: object, default: float, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _text(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _provider(value: object) -> str:
    provider = _text(value, "auto").lower()
    if provider in {"auto", "any", "default", "none"}:
        return "auto"
    if not re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", provider):
        raise ValueError("provider должен быть 'auto' или корректным slug OpenRouter")
    return provider


def _data_collection(value: object) -> str:
    policy = _text(value, "deny").lower()
    if policy not in {"allow", "deny"}:
        raise ValueError("provider_data_collection должен быть allow или deny")
    return policy


def _https_url(value: object, default: str, *, field: str) -> str:
    url = _text(value, default)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{field} должен быть HTTPS URL без встроенных учётных данных")
    return url


def _service_tier(value: object) -> str:
    tier = _text(value).lower()
    if tier in {"", "auto", "none"}:
        return ""
    if tier not in {"default", "flex", "priority"}:
        raise ValueError("service_tier должен быть default, flex или priority")
    return tier


def _vision_detail(value: object) -> str:
    detail = _text(value, "original").lower()
    if detail not in {"low", "high", "auto", "original"}:
        raise ValueError("vision_detail должен быть low, high, auto или original")
    return detail


def _reasoning_effort(value: object) -> str:
    effort = _text(value).lower()
    if effort in {"", "auto", "default"}:
        return ""
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError(
            "reasoning_effort должен быть none, minimal, low, medium, high или xhigh"
        )
    return effort


def _transcription_backend(value: object) -> str:
    backend = _text(value, "local").lower()
    if backend not in {"local", "openrouter"}:
        raise ValueError("transcription_backend должен быть local или openrouter")
    return backend


def _local_transcription_device(value: object) -> str:
    device = _text(value, "cuda").lower()
    if device not in {"cuda", "cpu", "auto"}:
        raise ValueError("local_transcription_device должен быть cuda, cpu или auto")
    return device


def _local_transcription_compute_type(value: object) -> str:
    compute_type = _text(value, "float16").lower()
    if compute_type not in {"default", "float16", "float32", "int8", "int8_float16"}:
        raise ValueError(
            "local_transcription_compute_type должен быть default, float16, float32, "
            "int8 или int8_float16"
        )
    return compute_type


def _fish_format(value: object) -> str:
    audio_format = _text(value, "wav").lower()
    if audio_format not in {"wav", "pcm", "mp3", "opus"}:
        raise ValueError("fish_format должен быть wav, pcm, mp3 или opus")
    return audio_format


@dataclass(frozen=True)
class Settings:
    project_root: Path
    openrouter_api_key: str
    openrouter_url: str
    openrouter_stt_url: str
    openrouter_referer: str
    openrouter_title: str
    model: str
    provider: str
    provider_allow_fallbacks: bool
    provider_data_collection: str
    provider_zdr: bool
    service_tier: str
    chat_timeout_seconds: float
    temperature: float
    reply_tokens: int
    history_turns: int
    vision_detail: str
    reasoning_effort: str
    transcription_backend: str
    transcription_model: str
    transcription_provider: str
    transcription_provider_allow_fallbacks: bool
    transcription_timeout_seconds: float
    transcription_language: str
    transcription_temperature: float
    transcription_normalize_audio: bool
    transcription_max_seconds: int
    local_transcription_model: str
    local_transcription_device: str
    local_transcription_compute_type: str
    local_transcription_beam_size: int
    local_transcription_hotwords: str
    local_transcription_wake_max_no_speech_prob: float
    local_transcription_wake_min_avg_logprob: float
    local_transcription_fallback_to_openrouter: bool
    local_transcription_cuda_path: str
    fish_api_key: str
    fish_voice_id: str
    fish_model: str
    fish_timeout_seconds: float
    fish_first_audio_timeout_seconds: float
    fish_emotion: str
    fish_max_chars: int
    fish_format: str
    microphone_device: str
    wake_phrases: tuple[str, ...]
    wake_aliases: tuple[str, ...]
    wake_model_path: str
    wake_followup_seconds: float
    vad_aggressiveness: int
    vad_silence_ms: int
    vad_max_seconds: int
    screen_monitor: int
    screen_max_width: int
    screen_max_height: int
    screen_original_for_text_requests: bool
    turn_timeout_seconds: float
    log_content: bool
    port: int
    max_request_bytes: int
    max_audio_bytes: int
    max_screen_bytes: int

    @property
    def fish_ready(self) -> bool:
        return bool(self.fish_api_key and self.fish_voice_id)

    def provider_preferences(
        self,
        provider: str,
        allow_fallbacks: bool,
    ) -> dict[str, object]:
        preferences: dict[str, object] = {
            "allow_fallbacks": allow_fallbacks,
            "data_collection": self.provider_data_collection,
        }
        if provider != "auto":
            preferences["order"] = [provider]
        if self.provider_zdr:
            preferences["zdr"] = True
        return preferences

    def validate_startup(self) -> None:
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY не задан. Скопируй .env.example в .env и заполни ключ."
            )
        if not self.wake_phrases:
            raise ValueError("wake_phrases не должен быть пустым")
        if self.transcription_backend == "local" and self.local_transcription_hotwords:
            raise ValueError(
                "local_transcription_hotwords должен оставаться пустым: подсказка имени "
                "провоцирует ложные wake-word"
            )


def load_settings(project_root: Path | None = None) -> Settings:
    default_root = PROJECT_ROOT if (PROJECT_ROOT / "config.yaml").exists() else Path.cwd()
    root = (project_root or default_root).resolve()
    load_dotenv(root / ".env", override=False)
    config_path = root / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml должен содержать YAML mapping/object")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("Все ключи config.yaml должны быть строками")
    excluded_fields = {
        "project_root",
        "openrouter_api_key",
        "fish_api_key",
        "fish_voice_id",
        "max_request_bytes",
        "max_audio_bytes",
        "max_screen_bytes",
    }
    known_fields = {field.name for field in fields(Settings)} - excluded_fields
    known_fields.update({"max_request_mb", "max_audio_mb", "max_screen_mb"})
    unknown_fields = sorted(set(raw) - known_fields)
    if unknown_fields:
        raise ValueError(f"Неизвестные поля config.yaml: {', '.join(unknown_fields)}")

    max_request_mb = _as_int(raw.get("max_request_mb"), 40, minimum=1, maximum=200)
    max_audio_mb = _as_int(raw.get("max_audio_mb"), 18, minimum=1, maximum=100)
    max_screen_mb = _as_int(raw.get("max_screen_mb"), 8, minimum=1, maximum=50)
    return Settings(
        project_root=root,
        openrouter_api_key=_text(os.environ.get("OPENROUTER_API_KEY")),
        openrouter_url=_https_url(
            os.environ.get("OPENROUTER_URL"),
            _text(raw.get("openrouter_url"), "https://openrouter.ai/api/v1/chat/completions"),
            field="openrouter_url",
        ),
        openrouter_stt_url=_https_url(
            os.environ.get("OPENROUTER_STT_URL"),
            _text(
                raw.get("openrouter_stt_url"),
                "https://openrouter.ai/api/v1/audio/transcriptions",
            ),
            field="openrouter_stt_url",
        ),
        openrouter_referer=_text(raw.get("openrouter_referer")),
        openrouter_title=_text(raw.get("openrouter_title"), "Berangaria Desktop Agent"),
        model=_text(
            os.environ.get("OPENROUTER_MODEL"),
            _text(raw.get("model"), "openai/gpt-5.6-luna"),
        ),
        provider=_provider(
            os.environ.get("OPENROUTER_PROVIDER", raw.get("provider", "openai"))
        ),
        provider_allow_fallbacks=_as_bool(raw.get("provider_allow_fallbacks"), True),
        provider_data_collection=_data_collection(raw.get("provider_data_collection")),
        provider_zdr=_as_bool(raw.get("provider_zdr"), False),
        service_tier=_service_tier(raw.get("service_tier", "priority")),
        chat_timeout_seconds=_as_float(
            raw.get("chat_timeout_seconds"), 60.0, minimum=5.0, maximum=180.0
        ),
        temperature=_as_float(raw.get("temperature"), 0.8, minimum=0.0, maximum=2.0),
        reply_tokens=_as_int(raw.get("reply_tokens"), 500, minimum=32, maximum=4096),
        history_turns=_as_int(raw.get("history_turns"), 12, minimum=1, maximum=50),
        vision_detail=_vision_detail(raw.get("vision_detail")),
        reasoning_effort=_reasoning_effort(raw.get("reasoning_effort", "none")),
        transcription_backend=_transcription_backend(raw.get("transcription_backend")),
        transcription_model=_text(raw.get("transcription_model"), "openai/whisper-large-v3"),
        transcription_provider=_provider(raw.get("transcription_provider", "auto")),
        transcription_provider_allow_fallbacks=_as_bool(
            raw.get("transcription_provider_allow_fallbacks"), True
        ),
        transcription_timeout_seconds=_as_float(
            raw.get("transcription_timeout_seconds"), 90.0, minimum=5.0, maximum=180.0
        ),
        transcription_language=_text(raw.get("transcription_language"), "ru").lower(),
        transcription_temperature=_as_float(
            raw.get("transcription_temperature"), 0.0, minimum=0.0, maximum=1.0
        ),
        transcription_normalize_audio=_as_bool(raw.get("transcription_normalize_audio"), True),
        transcription_max_seconds=_as_int(
            raw.get("transcription_max_seconds"), 60, minimum=5, maximum=600
        ),
        local_transcription_model=_text(
            raw.get("local_transcription_model"), "large-v3-turbo"
        ),
        local_transcription_device=_local_transcription_device(
            raw.get("local_transcription_device")
        ),
        local_transcription_compute_type=_local_transcription_compute_type(
            raw.get("local_transcription_compute_type")
        ),
        local_transcription_beam_size=_as_int(
            raw.get("local_transcription_beam_size"), 3, minimum=1, maximum=10
        ),
        local_transcription_hotwords=_text(raw.get("local_transcription_hotwords")),
        local_transcription_wake_max_no_speech_prob=_as_float(
            raw.get("local_transcription_wake_max_no_speech_prob"),
            0.35,
            minimum=0.0,
            maximum=1.0,
        ),
        local_transcription_wake_min_avg_logprob=_as_float(
            raw.get("local_transcription_wake_min_avg_logprob"),
            -1.30,
            minimum=-5.0,
            maximum=0.0,
        ),
        local_transcription_fallback_to_openrouter=_as_bool(
            raw.get("local_transcription_fallback_to_openrouter"), True
        ),
        local_transcription_cuda_path=_text(
            raw.get("local_transcription_cuda_path"), "runtime/cuda12"
        ),
        fish_api_key=_text(os.environ.get("FISH_API_KEY")),
        fish_voice_id=_text(os.environ.get("FISH_VOICE_ID")),
        fish_model=_text(raw.get("fish_model"), "s2.1-pro-free"),
        fish_timeout_seconds=_as_float(
            raw.get("fish_timeout_seconds"), 45.0, minimum=5.0, maximum=180.0
        ),
        fish_first_audio_timeout_seconds=_as_float(
            raw.get("fish_first_audio_timeout_seconds"),
            5.0,
            minimum=1.0,
            maximum=30.0,
        ),
        fish_emotion=_text(raw.get("fish_emotion"), "calm"),
        fish_max_chars=_as_int(raw.get("fish_max_chars"), 600, minimum=40, maximum=2000),
        fish_format=_fish_format(raw.get("fish_format")),
        microphone_device=_text(raw.get("microphone_device")),
        wake_phrases=tuple(
            phrase.strip().lower()
            for phrase in _text(raw.get("wake_phrases"), "бер,бэр,берангария,вер").split(",")
            if phrase.strip()
        ),
        wake_aliases=tuple(
            phrase.strip().lower()
            for phrase in _text(
                raw.get("wake_aliases"),
                "берт,бэрт,бед,биар,биэр,бир,биан,биаф,br",
            ).split(",")
            if phrase.strip()
        ),
        wake_model_path=_text(raw.get("wake_model_path"), "models/vosk-model-small-ru-0.22"),
        wake_followup_seconds=_as_float(
            raw.get("wake_followup_seconds"), 8.0, minimum=2.0, maximum=30.0
        ),
        vad_aggressiveness=_as_int(raw.get("vad_aggressiveness"), 2, minimum=0, maximum=3),
        vad_silence_ms=_as_int(raw.get("vad_silence_ms"), 900, minimum=300, maximum=3000),
        vad_max_seconds=_as_int(raw.get("vad_max_seconds"), 20, minimum=3, maximum=60),
        screen_monitor=_as_int(raw.get("screen_monitor"), 1, minimum=0, maximum=16),
        screen_max_width=_as_int(
            raw.get("screen_max_width"), 1920, minimum=320, maximum=7680
        ),
        screen_max_height=_as_int(
            raw.get("screen_max_height"), 1080, minimum=180, maximum=4320
        ),
        screen_original_for_text_requests=_as_bool(
            raw.get("screen_original_for_text_requests"), True
        ),
        turn_timeout_seconds=_as_float(
            raw.get("turn_timeout_seconds"), 45.0, minimum=10.0, maximum=180.0
        ),
        log_content=_as_bool(raw.get("log_content"), False),
        port=_as_int(raw.get("port"), 8765, minimum=0, maximum=65535),
        max_request_bytes=max_request_mb * 1024 * 1024,
        max_audio_bytes=max_audio_mb * 1024 * 1024,
        max_screen_bytes=max_screen_mb * 1024 * 1024,
    )
