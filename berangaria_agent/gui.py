"""Small visible control window for the background voice agent."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from berangaria_agent.background import configure_background_logging, run_background
from berangaria_agent.config import Settings

logger = logging.getLogger(__name__)

_STATE_LABELS = {
    "starting": "Запускаюсь",
    "listening": "Слушаю фразу «Бер»",
    "activating": "Проверяю обращение",
    "transcribing": "Распознаю речь",
    "heard": "Фраза распознана",
    "thinking": "Смотрю на экран и думаю",
    "reply": "Ответ готов",
    "speaking": "Говорю",
    "warning": "Предупреждение",
    "error": "Ошибка",
    "stopping": "Останавливаюсь",
    "stopped": "Остановлен",
}

_STATE_COLORS = {
    "starting": "#d99a24",
    "listening": "#38b26c",
    "activating": "#4a8fe7",
    "transcribing": "#4a8fe7",
    "heard": "#4a8fe7",
    "thinking": "#8b6ee8",
    "reply": "#38b26c",
    "speaking": "#db7dc6",
    "warning": "#d99a24",
    "error": "#d95757",
    "stopping": "#8a8f98",
    "stopped": "#8a8f98",
}


class AgentWindow:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = tk.Tk()
        self.root.title("Berangaria Agent")
        self.root.geometry("720x500")
        self.root.minsize(560, 400)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.closing = False

        self.status_text = tk.StringVar(value=_STATE_LABELS["starting"])
        self.detail_text = tk.StringVar(value="Подготавливаю микрофон и модель вызова…")
        self.request_text = tk.StringVar(value="—")
        self.latency_text = tk.StringVar(value="Полная задержка появится после первого ответа")

        self._build()
        self._set_state("starting", self.detail_text.get())
        self.root.after(100, self._drain_events)
        self.root.after(150, self._start_worker)

    def _build(self) -> None:
        self.root.configure(bg="#15171c")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#15171c")
        style.configure("Card.TFrame", background="#20232a")
        style.configure(
            "Title.TLabel",
            background="#15171c",
            foreground="#f4f4f5",
            font=("Segoe UI Semibold", 18),
        )
        style.configure(
            "Status.TLabel",
            background="#20232a",
            foreground="#f4f4f5",
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "Text.TLabel",
            background="#20232a",
            foreground="#b8bdc7",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Stop.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(16, 9),
        )

        root = ttk.Frame(self.root, style="Root.TFrame", padding=20)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Berangaria", style="Title.TLabel").pack(anchor="w")

        status_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        status_card.pack(fill="x", pady=(14, 12))
        status_row = ttk.Frame(status_card, style="Card.TFrame")
        status_row.pack(fill="x")
        self.indicator = tk.Canvas(
            status_row,
            width=18,
            height=18,
            bg="#20232a",
            highlightthickness=0,
        )
        self.indicator.pack(side="left", padx=(0, 10))
        self.indicator_dot = self.indicator.create_oval(3, 3, 15, 15, fill="#d99a24", outline="")
        ttk.Label(status_row, textvariable=self.status_text, style="Status.TLabel").pack(
            side="left"
        )
        ttk.Label(status_card, textvariable=self.detail_text, style="Text.TLabel").pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(status_card, textvariable=self.latency_text, style="Text.TLabel").pack(
            anchor="w", pady=(4, 0)
        )

        request_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        request_card.pack(fill="x", pady=(0, 12))
        ttk.Label(request_card, text="Последняя услышанная фраза", style="Text.TLabel").pack(
            anchor="w"
        )
        ttk.Label(request_card, textvariable=self.request_text, style="Status.TLabel").pack(
            anchor="w", pady=(6, 0)
        )

        reply_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        reply_card.pack(fill="both", expand=True)
        ttk.Label(reply_card, text="Последний ответ", style="Text.TLabel").pack(anchor="w")
        self.reply = tk.Text(
            reply_card,
            height=6,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            bg="#20232a",
            fg="#f4f4f5",
            insertbackground="#f4f4f5",
            font=("Segoe UI", 11),
            padx=0,
            pady=8,
        )
        self.reply.insert("1.0", "Пока ответов нет.")
        self.reply.configure(state="disabled")
        self.reply.pack(fill="both", expand=True)

        actions = ttk.Frame(root, style="Root.TFrame")
        actions.pack(fill="x", pady=(14, 0))
        ttk.Button(actions, text="Открыть лог", command=self._open_log).pack(side="left")
        self.stop_button = ttk.Button(
            actions,
            text="Остановить агента",
            style="Stop.TButton",
            command=self.stop,
        )
        self.stop_button.pack(side="right")

    def _start_worker(self) -> None:
        self.worker = threading.Thread(target=self._worker_main, daemon=True)
        self.worker.start()

    def _worker_main(self) -> None:
        try:
            asyncio.run(
                run_background(
                    self.settings,
                    stop_event=self.stop_event,
                    status_callback=lambda state, detail: self.events.put((state, detail)),
                )
            )
        except Exception as exc:
            logger.exception("Фоновый агент аварийно завершился")
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            self.events.put(("stopped", "Агент завершился из-за ошибки"))

    def _drain_events(self) -> None:
        try:
            while True:
                state, detail = self.events.get_nowait()
                self._set_state(state, detail)
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(100, self._drain_events)

    def _set_state(self, state: str, detail: str) -> None:
        if state == "metrics":
            self.latency_text.set(detail)
            return
        self.status_text.set(_STATE_LABELS.get(state, state))
        self.detail_text.set(detail or _STATE_LABELS.get(state, state))
        self.indicator.itemconfigure(self.indicator_dot, fill=_STATE_COLORS.get(state, "#8a8f98"))
        if state == "heard":
            self.request_text.set(detail)
        elif state == "reply":
            self.reply.configure(state="normal")
            self.reply.delete("1.0", "end")
            self.reply.insert("1.0", detail)
            self.reply.configure(state="disabled")
        elif state in {"warning", "error"}:
            stamp = datetime.now().strftime("%H:%M:%S")
            current = self.reply.get("1.0", "end").strip()
            message = f"[{stamp}] {detail}"
            if current and current != "Пока ответов нет.":
                message = f"{current}\n\n{message}"
            self.reply.configure(state="normal")
            self.reply.delete("1.0", "end")
            self.reply.insert("1.0", message)
            self.reply.configure(state="disabled")

    def _open_log(self) -> None:
        log_path = configure_background_logging(self.settings.project_root)
        os.startfile(log_path)

    def stop(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self._set_state("stopping", "Останавливаю микрофонный цикл…")

    def run(self) -> None:
        self.root.mainloop()


def run_gui(settings: Settings) -> None:
    AgentWindow(settings).run()
