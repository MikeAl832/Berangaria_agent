from types import SimpleNamespace

import pytest

from berangaria_agent.background import pcm_to_wav
from berangaria_agent.local_transcription import (
    LocalNoSpeechError,
    LocalTranscript,
    LocalWhisperTranscriber,
    configure_cuda_runtime,
)


class _FakeModel:
    def __init__(self, texts):
        self.texts = texts
        self.kwargs = None

    def transcribe(self, _audio, **kwargs):
        self.kwargs = kwargs
        return (
            [
                SimpleNamespace(
                    text=text,
                    start=0.0,
                    end=1.0,
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                )
                for text in self.texts
            ],
            SimpleNamespace(),
        )


def test_local_whisper_keeps_hotwords_disabled_and_sanitizes(settings):
    client = LocalWhisperTranscriber(settings)
    model = _FakeModel([" Бер, привет. ", " Продолжение следует..."])
    client._model = model

    result = client.transcribe(pcm_to_wav(b"\x01\x00" * 1600))

    assert result == "Бер, привет."
    assert model.kwargs["language"] == "ru"
    assert model.kwargs["hotwords"] is None
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["temperature"] == 0.0


def test_local_whisper_rejects_empty_or_noise_only_result(settings):
    client = LocalWhisperTranscriber(settings)
    client._model = _FakeModel([" *звук шума* "])

    with pytest.raises(LocalNoSpeechError):
        client.transcribe(pcm_to_wav(b"\x00\x00" * 1600))


def test_wake_only_confidence_filter(settings):
    confident = LocalTranscript("Бер", avg_logprob=-0.2, no_speech_probability=0.1)
    noisy_but_spoken = LocalTranscript("Бер", avg_logprob=-0.8, no_speech_probability=0.0)
    hallucination = LocalTranscript("Берангария", avg_logprob=-0.8, no_speech_probability=0.6)
    weak_audio = LocalTranscript("Бер", avg_logprob=-1.4, no_speech_probability=0.0)

    assert confident.reliable_as_wake_only(settings)
    assert noisy_but_spoken.reliable_as_wake_only(settings)
    assert not hallucination.reliable_as_wake_only(settings)
    assert not weak_audio.reliable_as_wake_only(settings)


def test_cuda_runtime_reports_missing_project_libraries(settings, monkeypatch):
    monkeypatch.setattr("berangaria_agent.local_transcription.shutil.which", lambda _name: None)
    with pytest.raises(Exception, match="CUDA 12/cuDNN 9"):
        configure_cuda_runtime(settings)
