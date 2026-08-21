"""CLI entry point: python -m berangaria_agent."""

from __future__ import annotations

import argparse
import asyncio

from berangaria_agent.background import input_devices, install_wake_model, run_background
from berangaria_agent.config import load_settings
from berangaria_agent.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальный голосовой агент Berangaria со снимками экрана"
    )
    parser.add_argument("--port", type=int, help="loopback-порт вместо config.yaml")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    parser.add_argument(
        "--background",
        action="store_true",
        help="фоновый голосовой режим без браузера",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="голосовой режим с небольшим окном состояния",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="показать доступные микрофоны и выйти",
    )
    parser.add_argument(
        "--install-wake-model",
        action="store_true",
        help="скачать локальную русскую модель распознавания фразы вызова",
    )
    args = parser.parse_args()
    if args.list_audio_devices:
        for device_id, name in input_devices():
            print(f"{device_id}: {name}")
        return
    if args.port is not None and not 0 <= args.port <= 65535:
        parser.error("port должен быть в диапазоне 0..65535")
    settings = load_settings()
    if args.install_wake_model:
        print(f"Wake-word модель готова: {install_wake_model(settings)}")
        return
    try:
        settings.validate_startup()
    except ValueError as exc:
        parser.error(str(exc))
    if args.background:
        try:
            asyncio.run(run_background(settings))
        except ValueError as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            print("\nBerangaria Agent остановлен")
        return
    if args.gui:
        from berangaria_agent.gui import run_gui

        run_gui(settings)
        return
    run_server(settings, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
