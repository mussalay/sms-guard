> Обновление 06.11.2026: рабочие языки только en/ru; казахский исключён по DEV-01 (docs/deviations.md). Описания исходного PDF ниже исторические; собственный набор — минимум100 ru, строго test-only. Русский ML не заявляется, правила и URL-анализ относятся к будущим этапам.

# SMS Guard

Интеллектуальная система обнаружения мошеннических SMS и подозрительных URL. Сейчас выполнены **задания 1–2**: анализ предметной области и запускаемый каркас. Анализ SMS/URL, авторизация, история, Dashboard и модели ещё не реализованы. Задания 3–18 требуют подтверждения пользователя. ML-метрик пока нет.

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
ml/artifacts/         # пока без моделей
risk/                 # пока без алгоритма
migrations/           # Alembic, пока без предметных revisions
templates/            # base.html, index.html
static/               # CSS и локальный Bootstrap с лицензией
tests/                # проверки каркаса
data/raw/             # пока пусто
data/processed/       # пока пусто
notebooks/            # пока пусто
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

Тесты работают с временными логами и SQLite in-memory. Проверяют health, честный статус моделей, локальный CSS, OpenAPI, отсутствие будущих API, SQLite/FK, настройки, экранирование и ротацию. Это не оценка ML и не полное тестирование будущих требований безопасности.

Приложение не логирует тела/URL запросов; сервер запускается без access-log, чтобы query string не попал в журналы. Логгер `sms_guard` пишет время, уровень, модуль и служебное сообщение; сейчас логируются старт/остановка и отсутствие моделей. Политика проекта: [AGENTS.md](AGENTS.md).

## Документация

- [Матрица требований](docs/requirements_matrix.md): все ФТ/НФТ и полный текст18 заданий с критериями.
- [План18 заданий и неоднозначности ТЗ](docs/implementation_plan.md).
- [Задание1: предметный анализ](docs/01_domain_analysis.md), [Use Case PlantUML](docs/diagrams/use_cases.puml).
- [Фактические проверки задания2](docs/02_verification.md).
- [Полный список созданных и сохранённых файлов](docs/created_files.md).
- [PDF47 страниц](docs/technical_specification.pdf), [извлечённый текст](docs/technical_specification.txt).

UML-исходник можно открыть в локальном PlantUML-редакторе. Качество языковой поддержки и ML-метрики будут указываться только после реальных экспериментов. Полный текст SMS сохраняется в будущем только с согласием; пользовательские URL никогда не открываются системой.
