# SMS Guard

Работающий локальный MVP: FastAPI + SQLAlchemy2 + Alembic + SQLite + Jinja2 + Bootstrap5 + Chart.js. Есть регистрация/вход/выход, реальные SMS- и URL-модели, Risk Score, результаты, приватная история, профиль и Dashboard.

## Запуск одной командой

Из корня проекта, где уже подготовлены `.venv` и локальные модели:

```powershell
.\.venv\Scripts\python.exe -m app
```

Адрес: **http://127.0.0.1:8080/**. Команда применяет миграции и запускает сервер без access-log; остановка Ctrl+C. Если сервер уже работает, просто откройте адрес. Docker не используется.

Демонстрационный аккаунт: **demo / DemoGuard2026!**. Это только локальные демонстрационные реквизиты; можно зарегистрировать свой аккаунт. `python -m app.seed_demo` добавляет до30 проверок из искусственных сообщений, рассчитанных моделями; уже существующие записи не удаляются. Для иного демо-пароля задайте SMS_GUARD_DEMO_PASSWORD до первого создания аккаунта.

## Подготовка окружения на другой машине

Python3.13; из корня:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m app
```

Не перезаписывайте существующий `.env`. Локальные joblib/metadata в `ml/artifacts/` и данные не коммитятся: при переносе нужно также перенести доверенные обученные артефакты. Без них приложение стартует, health честно сообщает false, анализ соответствующего типа возвращает503. Автоматического скачивания моделей при чистом Git-клонировании нет. Не загружайте недоверенные joblib. Шаблоны, Bootstrap и Chart.js локальные; Swagger `/docs` может требовать свои стандартные внешние ресурсы.

## Реальное качество и ограничения

| Test |Precision|Recall|F1|ROC-AUC|
|---|---:|---:|---:|---:|
|SMS: calibrated Linear SVM|0.9362|0.9462|0.9412|0.9960|
|URL: Random Forest|0.6425|0.7435|0.6893|0.8447|

SMS достиг целевых0.90. **URL-модель не достигла целей ТЗ**, возможны ложные тревоги на легитимные адреса. Вероятность не является гарантией безопасности.

Английский — основной язык SMS ML. Русский использует только правила и URL, P_sms=null; отдельный искусственный русский тест дал rules+URL Recall0.66 (не доказательство русского ML). Казахский исключён DEV-01. Русский тест никогда не использовался для обучения/подбора.

Подробный честный список сделанного и незакрытых критериев: [docs/completion.md](docs/completion.md). Рабочая функциональность создана по заданиям7–18; академическая приёмка всех пунктов ещё не закрыта. Исторические этапы1–6 не повторялись. Изменения последнего MAX SPEED этапа фиксируются одним итоговым коммитом.

## API

Отдельная страница **ML Analytics**: `/ml-analytics/` (после входа, ссылка в навигации).
Она показывает сохранённые test-метрики и матрицы ошибок SMS/URL, сравнения кандидатов
на validation, параметры, пороги, источники, размеры выборок и metadata.
Пользовательский Dashboard остаётся на `/`.

Read-only API: `GET /api/v1/ml/analytics/`, доступ только с действующей сессией.
Загрузчик читает фиксированный список JSON и проверяет согласованность артефактов;
нет обучения, inference, чтения сырых SMS/URL или пользовательских файловых путей.
При недоступных/несогласованных отчётах —503, без подставных метрик.
Метрики и confusion matrices рисуются Chart.js; ROC/PR показаны сохранёнными PNG
реальных validation-кривых. Числовых точек в JSON нет, кривые из AUC не восстанавливаются.
URL-график содержит LR C=1 и RF depth=12; таблица сравнивает все четыре конфигурации.
Сохраняются предупреждения о качестве URL и ограничениях языков.

Все изменения требуют cookie `sg_csrf` и совпадающий заголовок `X-CSRF-Token`. Получить cookie можно GET `/login/`; браузер использует meta-token автоматически. Для защищённых API требуется cookie-сессия; чужие записи дают404. Регистрация и вход также защищены CSRF.

| Метод и путь | JSON / назначение |
|---|---|
|GET /api/v1/health/|Liveness и bool загруженных моделей|
|POST /api/v1/auth/register/|username,email,password;201|
|POST /api/v1/auth/login/|username,password;200/401/429|
|POST /api/v1/auth/logout/|Уничтожение текущей сессии;204|
|POST /api/v1/check/sms/|text,1–1000символов;200/400/429/503|
|POST /api/v1/check/url/|url,до2048;200/400/429/503|
|GET /api/v1/checks/|date_from/date_to/type/category/risk_level/page;20настраницу|
|GET /api/v1/checks/{id}/|Свой подробный результат|
|DELETE /api/v1/checks/{id}/|Своё удаление;204|
|GET /api/v1/stats/summary/|days=7/30/90;scope=all только для администратора,без чужих текстов|
|GET/POST /api/v1/profile/|Согласие хранения;смена пароля с current_password,отзыв сессий|
|DELETE /api/v1/history/|Вся собственная история;204|

Для защиты повторов передавайте `Idempotency-Key` при анализе. Лимит30анализов/мин на пользователя;5ошибок входа дают60с паузы. Ограничители в памяти одного процесса: текущая конфигурация предназначена для локального запуска одним worker. Cookie HttpOnly/SameSite=Lax,Secure при HTTPS. Пароли PBKDF2+соль; ORM; CSRF; Jinja autoescape/textContent. Полные SMS не сохраняются без согласия, телефоны/карты/суммы маскируются; тексты и секреты не пишутся в логи.

Ссылки разбираются исключительно локально: нет HTTP/DNS/редиректов, PSL встроенный. Модели загружаются один раз при старте. Risk config — risk_config.yaml; граница medium22 подобрана на английской validation (исходная35),high70. Русский риск не проходил независимую калибровку.

## Проверки и воспроизводимость

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=ml --cov=risk --cov=app
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m alembic check
```

Итог:114passed,coverage78% всего/около70%ML/risk98%. Браузерные скриншоты: `docs/screenshots/`. Нагрузочные результаты: `docs/reports/mvp_acceptance.json`. Тесты работают со своей SQLite, не удаляют пользовательскую историю. Тесты реального inference требуют локальные модели; часть проверок без артефактов пропускается. Они не заменяют независимый аудит безопасности.

Готовые SMS split и FeatureUnion сохранены. `python -m ml.sms_experiment select` подбирает только train/validation, `verify` проверяет воспроизводимость без test, `finalize` эксклюзивно резервирует один финальный test. Повторный select/finalize после claim запрещён. URL-эксперимент: `python -m ml.url_experiment`, исходный архив должен быть в data/raw/phiusiil.zip. В текущем workspace финальные test уже использованы: не обходите claim и не подбирайте по этим данным. Для новых исследований нужен отдельный протокол/эксперимент, а не удаление отчётов.

`notebooks/02_models.ipynb` содержит реальные таблицы, ROC/PR/CM,анализ5FN/5FP,20URL и reload; он читает сохранённые test-предсказания вместо повторной оценки. Исполнение: `python scripts/run_notebook.py notebooks/02_models.ipynb`. В SVM calibration-fold векторизатор и scaler fit только на своём train-fold; рабочая модель не переобучалась на train+val после test.

## Структура и документы

- `app/main.py`, `app/web.py`: приложение, защищённые страницы/API.
- `app/core/security.py`, `app/db/models.py`, `migrations/`: сессии/пароли/БД.
- `app/analysis/services.py`, `social.py`: общий анализ и ru/en runtime-правила.
- `ml/sms_experiment.py`, `url_experiment.py`, `inference.py`, `url_features.py`, `risk.py`: настоящие модели и риск.
- `templates/app.html`, `static/js/app.js`, локальные vendor-ресурсы: интерфейс и графики.
- `ml/rules.py` не расширялся: это замороженный словарь признаков этапа6, чтобы не изменить уже оценённый SMS inference.
- [Итог и диаграммы](docs/completion.md), [матрица требований](docs/requirements_matrix.md), [план](docs/implementation_plan.md), [отклонения](docs/deviations.md), [все файлы](docs/created_files.md).
- [Исходный PDF](docs/technical_specification.pdf), [анализ предметной области](docs/01_domain_analysis.md), [Use Case](docs/diagrams/use_cases.puml), [историческая проверка3–6](docs/03_06_verification.md).
- [Паспорт UCI SMS](docs/datasets/uci_sms.md), [русский test-only](docs/datasets/custom_ru.md), паспорт PhiUSIIL — в итоговом документе.

Исторические отчёты описывают состояние своего этапа. Текущий статус и незакрытые пункты определяются docs/completion.md.
