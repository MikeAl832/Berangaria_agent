"""Tokenized loopback HTTP server for the local browser UI."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import json
import logging
import secrets
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from berangaria_agent.config import Settings
from berangaria_agent.service import AgentService, InputError, TurnRequest

logger = logging.getLogger(__name__)

_AUDIO_MIMES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
}
_SCREEN_MIMES = {"image/jpeg", "image/png", "image/webp"}


def _decode(value: object, *, field: str, maximum: int) -> bytes | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise InputError(f"{field} должен быть base64-строкой")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError(f"Некорректный base64 в {field}") from exc
    if len(decoded) > maximum:
        raise InputError(f"{field} превышает допустимый размер")
    return decoded or None


def _mime(value: object, *, fallback: str, allowed: set[str], field: str) -> str:
    if value in (None, ""):
        return fallback
    if not isinstance(value, str):
        raise InputError(f"{field} должен быть строкой")
    normalized = value.split(";", 1)[0].strip().lower()
    if normalized not in allowed:
        raise InputError(f"Неподдерживаемый {field}: {normalized}")
    return normalized


def parse_turn(payload: object, settings: Settings) -> TurnRequest:
    if not isinstance(payload, dict):
        raise InputError("Тело запроса должно быть JSON-объектом")
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise InputError("text должен быть строкой")
    if len(text) > 8_000:
        raise InputError("text длиннее 8000 символов")
    audio = _decode(
        payload.get("audio_base64"),
        field="audio_base64",
        maximum=settings.max_audio_bytes,
    )
    screen = _decode(
        payload.get("screen_base64"),
        field="screen_base64",
        maximum=settings.max_screen_bytes,
    )
    if not text.strip() and not audio:
        raise InputError("Нужен text или audio_base64")
    return TurnRequest(
        text=text,
        audio=audio,
        audio_mime=_mime(
            payload.get("audio_mime"),
            fallback="audio/webm",
            allowed=_AUDIO_MIMES,
            field="audio_mime",
        ),
        screen=screen,
        screen_mime=_mime(
            payload.get("screen_mime"),
            fallback="image/jpeg",
            allowed=_SCREEN_MIMES,
            field="screen_mime",
        ),
    )


class AgentHTTPServer(HTTPServer):
    service: AgentService
    settings: Settings
    token: str
    session_path: str
    origin: str


class RequestHandler(BaseHTTPRequestHandler):
    server: AgentHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("desktop-agent http: " + format, *args)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "media-src 'self' data: blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-Berangaria-Token", "")
        origin = self.headers.get("Origin")
        return hmac.compare_digest(supplied, self.server.token) and (
            origin is None or origin == self.server.origin
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        if path != self.server.session_path:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        index = Path(__file__).with_name("static") / "index.html"
        html = index.read_text(encoding="utf-8")
        html = html.replace("__BERANGARIA_TOKEN__", self.server.token)
        html = html.replace(
            "__CAPABILITIES__",
            json.dumps(
                {
                    "vision": True,
                    "voice_input": True,
                    "fish": self.server.settings.fish_ready,
                    "microphone_device": self.server.settings.microphone_device,
                }
            ),
        )
        self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")

    def _read_json(self) -> object:
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise InputError("Content-Type должен быть application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise InputError("Некорректный Content-Length") from exc
        if length <= 0 or length > self.server.settings.max_request_bytes:
            raise InputError("Некорректный или слишком большой запрос")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InputError("Некорректный JSON") from exc

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        path = urlsplit(self.path).path
        if path == "/api/reset":
            self.server.service.reset()
            self._json(HTTPStatus.OK, {"status": "reset"})
            return
        if path != "/api/turn":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            request = parse_turn(self._read_json(), self.server.settings)
            result = asyncio.run(self.server.service.process(request))
        except InputError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            logger.exception("Desktop agent turn failed")
            self._json(HTTPStatus.BAD_GATEWAY, {"error": f"Ошибка агента: {exc}"})
            return
        self._json(
            HTTPStatus.OK,
            {
                "transcript": result.transcript,
                "screen_description": result.screen_description,
                "reply": result.reply,
                "audio_base64": (
                    base64.b64encode(result.audio).decode("ascii") if result.audio else None
                ),
                "audio_mime": result.audio_mime,
                "warnings": list(result.warnings),
            },
        )


def run_server(settings: Settings, *, port: int | None = None, open_browser: bool = True) -> None:
    token = secrets.token_urlsafe(32)
    httpd = AgentHTTPServer(("127.0.0.1", settings.port if port is None else port), RequestHandler)
    actual_port = int(httpd.server_address[1])
    httpd.settings = settings
    httpd.service = AgentService(settings)
    httpd.token = token
    httpd.session_path = f"/session/{token}"
    httpd.origin = f"http://127.0.0.1:{actual_port}"
    url = f"{httpd.origin}{httpd.session_path}"
    print(f"Berangaria Agent: {url}")
    print("Доступ только с этого компьютера. Остановка: Ctrl+C")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nBerangaria Agent остановлен")
    finally:
        httpd.server_close()
