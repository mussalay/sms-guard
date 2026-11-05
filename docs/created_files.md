# Созданные и сохранённые файлы

Обновлено06.11.2026 после заданий3–6. Исходный PDF с русским именем и его идентичная копия сохранены.

Основные новые исходники:

- `ml/data/load_sms.py`
- `ml/data/custom_ru.py`
- `ml/preprocessing.py`
- `ml/eda.py`
- `ml/features.py`
- `ml/rules.py`
- `ml/data/split_sms.py`
- `ml/prepare_features.py`
- `notebooks/01_eda_sms.ipynb`
- `scripts/build_eda_notebook.py`
- `scripts/run_notebook.py`

Локальные генерируемые файлы вне Git: `.venv/`, `.env`, `sms_guard.db`, логи/кеши; `data/raw/sms_spam_collection.zip`, `SMSSpamCollection`, `readme`, `nltk_stopwords.zip`; `data/processed/sms_raw.csv`, `sms_clean.csv`, `sms_split.json`, `X_train.npz`, `X_val.npz`, `X_test.npz`, `y_train.npy`, `y_val.npy`, `y_test.npy`, `sms_feature_names.json`; `ml/artifacts/sms_feature_pipeline.joblib`. Никакого sms_model.joblib нет.

Русский `data/raw/custom_ru.csv` отслеживается как test-only исключение. Полная приёмка — `03_06_verification.md`.

Полный список отслеживаемых/новых файлов:

```text
.env.example
.gitattributes
.gitignore
AGENTS.md
README.md
alembic.ini
app/__init__.py
app/accounts/__init__.py
app/analysis/__init__.py
app/api/__init__.py
app/api/v1/__init__.py
app/api/v1/health.py
app/core/__init__.py
app/core/config.py
app/core/logging.py
app/dashboard/__init__.py
app/db/__init__.py
app/db/session.py
app/main.py
data/processed/.gitkeep
data/raw/.gitkeep
data/raw/custom_ru.csv
docs/01_domain_analysis.md
docs/02_verification.md
docs/03_06_verification.md
docs/03_data_collection.md
docs/04_preprocessing.md
docs/05_eda.md
docs/06_feature_results.md
docs/06_text_features.md
docs/created_files.md
docs/datasets/custom_ru.md
docs/datasets/uci_sms.md
docs/datasets/uci_source.json
docs/deviations.md
docs/diagrams/.gitkeep
docs/diagrams/use_cases.puml
docs/figures/05_class_balance.png
docs/figures/05_entity_rates.png
docs/figures/05_message_lengths.png
docs/figures/05_top_bigrams.png
docs/figures/05_top_words.png
docs/implementation_plan.md
docs/reports/03_sms_loading.json
docs/reports/04_preprocessing.json
docs/reports/05_eda.json
docs/reports/06_features.json
docs/requirements_matrix.md
docs/technical_specification.pdf
docs/technical_specification.txt
docs/ТЗ_Интеллектуальная_система_обнаружения_мошеннических_SMS_и_подозрительных (1).pdf
logs/.gitkeep
migrations/env.py
migrations/script.py.mako
migrations/versions/.gitkeep
ml/__init__.py
ml/artifacts/.gitkeep
ml/data/__init__.py
ml/data/custom_ru.py
ml/data/load_sms.py
ml/data/split_sms.py
ml/eda.py
ml/features.py
ml/prepare_features.py
ml/preprocessing.py
ml/resources/NLTK_STOPWORDS_README
ml/resources/provenance.json
ml/resources/stopwords_en.txt
ml/resources/stopwords_ru.txt
ml/rules.py
notebooks/.gitkeep
notebooks/01_eda_sms.ipynb
pyproject.toml
requirements.txt
risk/__init__.py
scripts/build_eda_notebook.py
scripts/run_notebook.py
static/css/app.css
static/vendor/.gitkeep
static/vendor/BOOTSTRAP-LICENSE
static/vendor/bootstrap.min.css
templates/base.html
templates/index.html
tests/conftest.py
tests/test_data_loading.py
tests/test_eda.py
tests/test_features.py
tests/test_preprocessing.py
tests/test_scaffold.py
```
