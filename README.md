# Berangaria Agent

[![Windows CI](https://github.com/MikeAl832/Berangaria_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/MikeAl832/Berangaria_agent/actions/workflows/ci.yml)

Berangaria Agent — самостоятельный голосовой ассистент для Windows. Он работает
в фоне без чата: слушает локальный микрофон, активируется по имени, получает один
текущий снимок экрана и отвечает голосом.

Проект не зависит от Telegram-бота Berangaria и не импортирует его код.

## Что уже работает

- видимый GUI со статусом прослушивания, распознавания, модели и озвучивания;
- локальные VAD и `faster-whisper large-v3-turbo` на GPU;
- вызов словами «Бер», «Бэр» или «Берангария» и безопасными фонетическими
  вариантами Whisper;
- защита от ложных wake-word на тишине, шуме и обычном разговоре;
- один PNG-снимок выбранного монитора на принятый запрос;
- нативный vision GPT-5.6 Luna через OpenRouter;
- короткая история разговора только в RAM;
- потоковая PCM-озвучка Fish Audio;
- раздельные метрики VAD, STT, снимка, Luna, Fish и времени до первого звука;
- диагностический web-интерфейс на `127.0.0.1`.

Схема компонентов и поток данных описаны в
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Требования

- Windows 10 или 11;
- Python 3.11 либо [uv](https://docs.astral.sh/uv/);
- микрофон;
- ключ OpenRouter;
- ключ и `voice_id` Fish Audio для основного голосового режима;
- NVIDIA GPU для быстрой локальной транскрипции — необязательно, доступен CPU
  или облачный fallback.

Для GPU-режима актуальный `faster-whisper` требует CUDA 12 и cuDNN 9. Это
зафиксировано в [официальной документации faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu),
а установка cuDNN на Windows описана в
[документации NVIDIA](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html).

## Быстрый запуск

```powershell
git clone https://github.com/MikeAl832/Berangaria_agent.git
cd Berangaria_agent
.\start-agent.bat
```

При первом запуске BAT-файл:

1. создаст `.venv` с Python 3.11;
2. установит зависимости;
3. создаст `.env` и `config.yaml` из безопасных примеров;
4. откроет `.env` в Блокноте;
5. запустит основной GUI.

Заполни `.env`:

```env
OPENROUTER_API_KEY=...
FISH_API_KEY=...
FISH_VOICE_ID=...
```

Секреты, пользовательский `config.yaml`, модели, CUDA DLL и логи исключены из
Git.

После первого успешного запуска можно использовать `start-gui.bat`: он открывает
только небольшое окно агента без консоли.

## Локальный Whisper

По умолчанию используется NVIDIA GPU:

```yaml
transcription_backend: "local"
local_transcription_device: "cuda"
local_transcription_compute_type: "float16"
local_transcription_cuda_path: "runtime/cuda12"
```

Помести CUDA 12/cuDNN 9 DLL в `runtime/cuda12` либо установи их в системный
`PATH`. Минимально проверяются `cublas64_12.dll` и `cudnn64_9.dll`; остальные
зависимые DLL должны находиться рядом. Сама модель Whisper скачивается в
`models/faster-whisper` при первом запуске.

CPU-вариант медленнее, но не требует NVIDIA:

```yaml
local_transcription_device: "cpu"
local_transcription_compute_type: "int8"
```

Если local Whisper недоступен и включён
`local_transcription_fallback_to_openrouter`, агент автоматически подготовит
небольшую Vosk-модель вызова, а распознавание принятой фразы выполнит через
OpenRouter. Модель fallback можно скачать заранее:

```powershell
.\.venv\Scripts\python.exe -m berangaria_agent --install-wake-model
```

## Использование

Скажи одной фразой:

```text
Бер, что сейчас на экране?
```

Если сказать только «Бер», прозвучит системный сигнал, после которого следующая
фраза в течение восьми секунд будет принята как запрос.

Основные варианты имени перечислены в `wake_phrases`. Ошибочные написания вроде
«Берт» и «Биар» находятся отдельно в `wake_aliases` и срабатывают только первым
словом — это уменьшает риск активации внутри обычного разговора. Не следует
включать имя в `local_transcription_hotwords`: такая подсказка провоцирует
галлюцинации wake-word на шуме.

Полезные команды:

```powershell
# Основной видимый GUI
.\.venv\Scripts\python.exe -m berangaria_agent --gui

# Фоновый режим в консоли
.\.venv\Scripts\python.exe -m berangaria_agent --background

# Доступные устройства записи
.\.venv\Scripts\python.exe -m berangaria_agent --list-audio-devices

# Диагностический web-интерфейс
.\.venv\Scripts\python.exe -m berangaria_agent
```

Пустой `microphone_device` в `config.yaml` выбирает системный микрофон Windows.
Для явного выбора скопируй точное название из `--list-audio-devices`.

## Экран и vision

На каждый принятый голосовой запрос снимается ровно один текущий кадр, а не
непрерывное видео. Обычный кадр уменьшается качественным Lanczos-фильтром до
`screen_max_width` × `screen_max_height`. Для просьб прочитать текст, код,
терминал или ошибку может отправляться исходное разрешение, если включено
`screen_original_for_text_requests`.

Модель получает запрос и кадр одним мультимодальным сообщением. Текст на экране
помечается как недоверенное наблюдение и не может менять системные инструкции.

## Приватность и безопасность

- VAD, local Whisper и проверка имени работают на компьютере.
- Посторонний разговор транскрибируется локально, но не отправляется в облако.
- После вызова OpenRouter получает текст запроса и один снимок экрана.
- Fish Audio получает только текст готового ответа.
- При аварийном переходе на OpenRouter STT аудио принятой wake-word фразы уходит
  в OpenRouter.
- История хранится только в памяти процесса и исчезает после остановки.
- Локальный web-сервер слушает только `127.0.0.1`, использует случайный токен и
  same-origin проверку.
- Агент не кликает, не печатает и не управляет операционной системой.

## Конфигурация

Публикуемый шаблон находится в [`config.example.yaml`](config.example.yaml).
Рабочий `config.yaml` создаётся локально и не отслеживается Git. Основные группы:

- `local_transcription_*` — модель, устройство, CUDA и фильтры уверенности;
- `wake_*` и `vad_*` — активация и завершение записи;
- `screen_*` — монитор и размер кадра;
- `model`, `provider`, `service_tier` — маршрут Luna;
- `fish_*` — голос и потоковая озвучка.

Рабочий лог записывается в `berangaria-agent.log` с ротацией. Он содержит
транскрипты, ответы и задержки, поэтому не публикуется в Git.

## Разработка и проверки

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m bandit -q -ll -r berangaria_agent
cmd.exe /d /c "call start-agent.bat --help"
```

Windows CI выполняет тесты, Ruff, Bandit и сборку пакета на Python 3.11.

## Ограничения и следующие шаги

1. Barge-in: остановка текущей озвучки, когда владелец начинает говорить.
2. System tray и настраиваемый автозапуск.
3. Отдельный обученный keyword-spotter вместо фонетических алиасов Whisper.
4. Снимок только при заметном изменении кадра.
5. Персистентная память с явной политикой хранения и удаления.
6. Управление компьютером как отдельный контур с allowlist, подтверждением и
   журналом действий.
