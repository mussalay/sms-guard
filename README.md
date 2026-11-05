# SMS Guard

Интеллектуальная система обнаружения мошеннических SMS и подозрительных URL. Выполнены **задания1–6**: документация, FastAPI-каркас, оригинальные UCI-данные, отдельный русский тест, нормализация/дедупликация, выполненный EDA notebook и воспроизводимый pipeline признаков. Классификаторы не обучались, ML-метрик нет. **Задание7 и далее — только после подтверждения пользователя.** Авторизация, анализ через веб, история и Dashboard ещё не реализованы.

Изменение06.11.2026: рабочие языки en/ru, казахский исключён [DEV-01](docs/deviations.md). Основной корпус — английский UCI. Русский набор100искусственных SMS используется исключительно как отдельный будущий тест, не для обучения/EDA/настройки. Русский ML не заявляется; пока предусмотрены словарь выражений и будущий URL-анализ. Небольшой en/ru словарь сейчас используется только для ручного признака, URL-модуль ещё не реализован.

Python 3.13, FastAPI, SQLAlchemy 2, Alembic, SQLite, Jinja2, Bootstrap 5; установлены scikit-learn, pandas, NumPy и NLTK для будущих этапов. Chart.js подключается в задании17. Разрешение альтернативы Django: ТЗ стр.12. SQLite подходит локальной разработке и учебному запуску без отдельного сервера; конкурентные записи ограничены, нагрузочные цели ещё не проверены. Docker не нужен.

## Запуск в PowerShell

Из корня полученного репозитория, при установленном Python3.13, семь команд:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --no-access-log
```

Не перезаписывайте существующий `.env` при повторном запуске. Активация venv не требуется. Остановка сервера — Ctrl+C. Для разработки можно добавить `--reload`. Секрет пока не используется, перед будущей реализацией сессий потребуется локально сгенерированный ключ. Не использовать каркас как готовый защищённый продукт.

Откройте [главную](http://127.0.0.1:8080/), [health](http://127.0.0.1:8080/api/v1/health/), [Swagger](http://127.0.0.1:8080/docs). Health:

```json
{"status":"ok","models":{"sms":false,"url":false}}
```

`ok` означает работающий каркас, оба `false` означают отсутствие подключённых моделей. Другие API не создавались. Главная пока публичная заглушка; защита будущего Dashboard относится к заданию14. Bootstrap5.3.8 хранится локально с MIT-лицензией, CDN для главной не нужен. Swagger использует стандартные внешние ресурсы FastAPI и может потребовать интернет.

## Структура

```text
app/
  main.py             # app factory, lifespan, Jinja2, static
  core/               # Settings, logging
  api/v1/health.py
  db/                 # DeclarativeBase, SQLite engine, session factory
  accounts/           # пустые пакеты будущих модулей
  analysis/
  dashboard/
ml/                   # preprocessing, EDA, feature pipeline, rules, data tools
ml/artifacts/         # feature pipeline локально; классификаторов нет
risk/                 # пока без алгоритма
migrations/           # Alembic, пока без предметных revisions
templates/            # base.html, index.html
static/               # CSS и локальный Bootstrap с лицензией
tests/                # каркас, загрузка, нормализация, EDA, split и признаки
data/raw/             # оригинал UCI локально + отслеживаемый synthetic custom_ru.csv
data/processed/       # CSV, split, разреженные матрицы и target; вне Git
notebooks/            # выполненный 01_eda_sms.ipynb
logs/                 # app.log + 5 backups по1МБ, исключены из Git
docs/                 # ТЗ, анализ, диаграммы, матрица, план, отчёт
```

Alembic использует `SMS_GUARD_DATABASE_URL`, как приложение; сейчас `upgrade head` проверяет инфраструктуру и создаёт служебную таблицу версии, предметные таблицы не создаёт. SQLAlchemy включает foreign_keys для SQLite. В `.env.example` перечислены настройки с префиксом `SMS_GUARD_`. Пути шаблонов и статики привязаны к корню проекта; относительный database URL из `.env` разрешается от текущего каталога, поэтому запускать из корня.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q app ml risk migrations tests
```

Тесты работают с временными файлами и SQLite in-memory. Помимо каркаса проверяют входные данные, нормализацию, дубликаты/конфликты, графики, split, отсутствие утечки TF-IDF, запрет доступа к русскому тесту и повторную генерацию матриц. Сетевые скачивания в тестах подменены; интернет и оригинальный UCI для pytest не требуются. Это не оценка классификатора и не полная проверка будущих функций безопасности.

Приложение не логирует тела/URL запросов; сервер запускается без access-log, чтобы query string не попал в журналы. Логгер `sms_guard` пишет время, уровень, модуль и служебное сообщение; сейчас логируются старт/остановка и отсутствие моделей. Политика проекта: [AGENTS.md](AGENTS.md).

## Воспроизведение заданий3–6

После установки requirements.txt, из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m ml.data.load_sms --download
.\.venv\Scripts\python.exe -m ml.data.custom_ru
.\.venv\Scripts\python.exe -m ml.preprocessing
.\.venv\Scripts\python.exe scripts/run_notebook.py
.\.venv\Scripts\python.exe -m ml.prepare_features
.\.venv\Scripts\python.exe -m pytest --cov=ml --cov=risk --cov-report=term-missing
```

Первый шаг скачивает только фиксированный официальный ZIP UCI с timeout30с; повторно проверяет SHA-256 и использует кеш. Ошибка загрузки/изменение архива не заменяются синтетическими обучающими данными. ZIP/SMSSpamCollection/readme сохраняются без изменений. Подготовка всех следующих этапов работает без сети. Нормализация использует локальные en/ru стоп-слова NLTK с provenance; никакие адреса из корпуса не открываются.

Исходный UCI:5574(4827ham/747spam). После очистки5135(4517/618), удалены403точных+36нормализованных повторов. Split seed42:3594train/770val/771test, по классам3161/433,678/92,678/93. Матрицы59924столбца:8785word+51133char+6manual. Pipeline fit только train; val/test только transform. Локальные файлы данных/артефактов исключены из Git; для повторения на другой машине сначала выполнить команды выше. CSV русского теста отслеживается как явное исключение.

Открыть notebook интерактивно: `python -m notebook notebooks/01_eda_sms.ipynb` из активированного venv. Скрипт исполнения автоматически создаёт локальный kernelspec текущего Python в `.cache/jupyter`; вручную выбранное ядро должно использовать то же окружение. Для восстановления исходных ячеек: `python scripts/build_eda_notebook.py`, затем повторить исполнение (builder сбрасывает outputs). В Windows Jupyter может выводить служебное предупреждение о TCP-транспорте локального ядра; ошибки ячеек считаются ошибкой выполнения.

EDA описывает весь очищенный корпус до обучения без подбора параметров (разрешено критерием задания5). Топ-15 TF-IDF задания6 вычисляется только на train. Сохранённый feature pipeline не является классификатором и не меняет `health.models=false`. Открывать joblib можно только из доверенного локального источника.

## Документация

- [Матрица требований](docs/requirements_matrix.md): все ФТ/НФТ и полный текст18 заданий с критериями.
- [План18 заданий и неоднозначности ТЗ](docs/implementation_plan.md).
- [Задание1: предметный анализ](docs/01_domain_analysis.md), [Use Case PlantUML](docs/diagrams/use_cases.puml).
- [Фактические проверки задания2](docs/02_verification.md).
- [Отчёт заданий3–6 и приёмка](docs/03_06_verification.md).
- [Загрузка](docs/03_data_collection.md), [нормализация](docs/04_preprocessing.md), [EDA](docs/05_eda.md), [признаки](docs/06_text_features.md), [размеры и топ-15](docs/06_feature_results.md).
- [Паспорт UCI](docs/datasets/uci_sms.md), [русский test-only](docs/datasets/custom_ru.md), [выполненный notebook](notebooks/01_eda_sms.ipynb).
- [Полный список созданных и сохранённых файлов](docs/created_files.md).
- [PDF47 страниц](docs/technical_specification.pdf), [извлечённый текст](docs/technical_specification.txt).

UML-исходник можно открыть в локальном PlantUML-редакторе. Качество языковой поддержки и ML-метрики будут указываться только после реальных экспериментов. Полный текст SMS сохраняется в будущем только с согласием; пользовательские URL никогда не открываются системой.
