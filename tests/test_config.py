from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from berangaria_agent.config import load_settings


def test_published_config_example_is_valid_and_machine_neutral():
    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))

    assert isinstance(raw, dict)
    assert raw["microphone_device"] == ""
    assert raw["local_transcription_hotwords"] == ""


def test_load_settings_reads_yaml_and_environment(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "model: yaml/model\nport: 9000\nhistory_turns: 7\nprovider: auto\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "env/model")
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)

    settings = load_settings(tmp_path)

    assert settings.model == "env/model"
    assert settings.port == 9000
    assert settings.history_turns == 7
    assert settings.vision_detail == "original"
    assert settings.reasoning_effort == ""
    assert settings.service_tier == ""
    assert settings.transcription_backend == "local"
    assert settings.transcription_model == "openai/whisper-large-v3"
    assert settings.transcription_provider == "auto"
    assert settings.transcription_provider_allow_fallbacks is True
    assert settings.transcription_language == "ru"
    assert settings.transcription_temperature == 0.0
    assert settings.transcription_normalize_audio is True
    assert settings.local_transcription_model == "large-v3-turbo"
    assert settings.local_transcription_device == "cuda"
    assert settings.local_transcription_compute_type == "float16"
    assert settings.local_transcription_fallback_to_openrouter is True
    assert settings.screen_max_width == 1920
    assert settings.screen_max_height == 1080
    assert settings.screen_original_for_text_requests is True
    assert settings.fish_format == "wav"
    assert settings.fish_ready is False


def test_startup_requires_only_openrouter(settings):
    minimal = replace(
        settings,
        fish_api_key="",
        fish_voice_id="",
    )
    minimal.validate_startup()


def test_startup_missing_openrouter_has_actionable_error(settings):
    missing = replace(settings, openrouter_api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        missing.validate_startup()


def test_load_settings_uses_built_in_defaults_without_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    settings = load_settings(tmp_path)

    assert settings.model == "openai/gpt-5.6-luna"
    assert settings.transcription_model == "openai/whisper-large-v3"
    assert settings.fish_format == "wav"


def test_load_settings_preserves_explicit_reasoning_none(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "reasoning_effort: none\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    settings = load_settings(tmp_path)

    assert settings.reasoning_effort == "none"
