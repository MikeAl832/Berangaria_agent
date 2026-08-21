import base64

import pytest

from berangaria_agent.server import parse_turn
from berangaria_agent.service import InputError


def test_parse_turn_decodes_and_normalizes_media(settings):
    request = parse_turn(
        {
            "audio_base64": base64.b64encode(b"audio").decode("ascii"),
            "audio_mime": "audio/webm;codecs=opus",
            "screen_base64": base64.b64encode(b"screen").decode("ascii"),
            "screen_mime": "image/jpeg",
        },
        settings,
    )
    assert request.audio == b"audio"
    assert request.audio_mime == "audio/webm"
    assert request.screen == b"screen"
    assert request.screen_mime == "image/jpeg"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "Нужен text"),
        ({"text": 42}, "text должен быть строкой"),
        ({"text": "x", "screen_base64": "%%%"}, "Некорректный base64"),
        ({"text": "x", "screen_mime": "image/svg+xml"}, "Неподдерживаемый"),
    ],
)
def test_parse_turn_rejects_bad_input(settings, payload, message):
    with pytest.raises(InputError, match=message):
        parse_turn(payload, settings)
