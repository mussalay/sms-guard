# 03. Карта исходного кода: 38 точек технической защиты

Номера строк извлечены из текущих файлов автоматически; исходный код при подготовке не менялся. Версия реализации —36 cec01, аудит —docs/final_audit.md. **Файл → функция → вход → результат → вызовы → смысл → показ** составляют маршрут ответа. В VS Code Ctrl+P открывает путь, Ctrl+G принимает номер строки, Ctrl+F находит имя. Показ исходника не требует запуска обучения. Английские идентификаторы сохранены как в коде; их смысл раскрывается по-русски.

## 1. Загрузка SMS-датасета

**Путь и строки:** [ml/data/load_sms.py:32](../../ml/data/load_sms.py#L32), диапазон 32–77. **Объект:** `download_uci`.
**Вход (сигнатура):** `raw_dir: Path=RAW_DIR, record_path: Path=SOURCE_RECORD`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** Path исходного SMSSpamCollection; load_sms отдельно возвращает DataFrame(id, text, label).
**Вызовы в этом объекте:** `ValueError, ZoneInfo, archive.exists, archive.read_bytes, archive.stat, archive.write_bytes, datetime.fromtimestamp, datetime.fromtimestamp(archive.stat().st_mtime, ZoneInfo('Asia/Qyzylorda')).isoformat, io.BytesIO, json.loads, len, raw_dir.mkdir, record_path.exists, record_path.read_text, response.read, set, sha256, target.exists, target.read_bytes, target.write_bytes, urlopen, write_json, zipfile.ZipFile, zipped.namelist, zipped.read`.
**Ключевой смысл:** SOURCE_URL фиксирован на официальном UCI; SHA проверяет неизменность, строгий UTF-8 и partition по первой tab сохраняют текст. prepare_raw сохраняет CSV/паспортную статистику.

```text
41: raise ValueError("Архив UCI превышает допустимый размер")
44: raise ValueError("Неожиданный состав архива UCI")
50: raise ValueError("SHA256 архива не совпадает с паспортом; источник мог измениться")
74: raise ValueError(f"Исходный файл {name} изменён; перезапись запрещена")
```

**Быстрый поиск:** Ctrl+P → `ml/data/load_sms.py`; Ctrl+G → `32`; Ctrl+F → `download_uci`.
**Показ преподавателю:** Показать константу источника, hash guard и load_sms; загрузчик сейчас не запускать.

## 2. Загрузка URL-датасета

**Путь и строки:** [ml/url_experiment.py:22](../../ml/url_experiment.py#L22), диапазон 22–160. **Объект:** `run`.
**Вход (сигнатура):** `root: Path=ROOT`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict отчёта и файловые артефакты; отдельного downloader нет.
**Вызовы в этом объекте:** `(root / 'docs/reports/url_test_claim.json').exists, (root / 'docs/reports/url_test_claim.json').open, GroupShuffleSplit, GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42).split, LogisticRegression, RandomForestClassifier, RuntimeError, StandardScaler, archive.read_bytes, choose_threshold, features.groupby, features.groupby('label')[FEATURE_NAMES[:6]].mean, features.groupby('label')[FEATURE_NAMES[:6]].mean().to_dict, features.to_parquet, float, frame.canonical.isin, frame.canonical.map, frame.drop_duplicates, frame.dropna, frame.dropna().copy, frame.groupby, frame.groupby('canonical').label.nunique, frame.label.astype, frame.label.to_numpy, frame.reset_index, groups.tolist, int, joblib.dump, json.dump, json.dumps, len, make_pipeline, max, metrics, min, model.fit, model.predict_proba, model_features, model_url, n.endswith, next, normalized.append, now, np.asarray, np.quantile, parts.items, pd.DataFrame, pd.read_csv, print, report_path.exists, rows.append, set, sha256, splitter.split, time.perf_counter, train_test_split, v.tolist, write_json, y[v].sum, zipfile.ZipFile, zipped.namelist, zipped.open`.
**Ключевой смысл:** Оригинальный ZIP был загружен вручную с официального UCI. run читает локальный phiusiil.zip через ZipFile и pd.read_csv(usecols=[URL, label]); 1-label инвертирует смысл. Автоматическую HTTP-загрузку этой функции не приписывать.

```text
25: raise RuntimeError("URL test уже оценён: повторный подбор запрещён")
111: model.fit(X[train], y[train])
112: p = model.predict_proba(X[val])[:, 1]
122: pval = model.predict_proba(X[val])[:, 1]
```

**Быстрый поиск:** Ctrl+P → `ml/url_experiment.py`; Ctrl+G → `22`; Ctrl+F → `run`.
**Показ преподавателю:** Показать строку archive и usecols; ссылку/лицензию/дату взять из url_experiment.json. Не выполнять run.

## 3. Очистка и нормализация SMS

**Путь и строки:** [ml/preprocessing.py:83](../../ml/preprocessing.py#L83), диапазон 83–89. **Объект:** `normalize_sms`.
**Вход (сигнатура):** `text: str, *, remove_stopwords: bool=False, language: str='en'`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** str нормализованного текста; clean_sms возвращает (DataFrame, dict отчёта).
**Вызовы в этом объекте:** `' '.join, TOKENS.tokenize, load_stopwords, normalization_steps`.
**Ключевой смысл:** normalize_sms вызывает normalization_steps, где порядок email/URL/money/phone/number защищает вложенные числа. clean_sms отдельно удаляет пустые и повторы до split.

```text
89: return cleaned
```

**Быстрый поиск:** Ctrl+P → `ml/preprocessing.py`; Ctrl+G → `83`; Ctrl+F → `normalize_sms`.
**Показ преподавателю:** Проследить один искусственный SMS вручную от NFKC до whitespace; не запускать main подготовки данных.

## 4. Токенизация

**Путь и строки:** [ml/preprocessing.py:83](../../ml/preprocessing.py#L83), диапазон 83–89. **Объект:** `normalize_sms`.
**Вход (сигнатура):** `text: str, *, remove_stopwords: bool=False, language: str='en'`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** str после опционального RegexpTokenizer; word TF-IDF сам выделяет свои токены.
**Вызовы в этом объекте:** `' '.join, TOKENS.tokenize, load_stopwords, normalization_steps`.
**Ключевой смысл:** TOKENS задан в начале модуля NLTK RegexpTokenizer. Он используется при remove_stopwords=True. Word token_pattern и char_wb находятся в features.py и не совпадают с общим tokenizer.

```text
89: return cleaned
```

**Быстрый поиск:** Ctrl+P → `ml/preprocessing.py`; Ctrl+G → `83`; Ctrl+F → `normalize_sms`.
**Показ преподавателю:** Показать TOKENS и условную ветку; объяснить сохранение <url> и различие word/char.

## 5. Удаление стоп-слов

**Путь и строки:** [ml/preprocessing.py:32](../../ml/preprocessing.py#L32), диапазон 32–36. **Объект:** `load_stopwords`.
**Вход (сигнатура):** `language: str='en'`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** frozenset строк en/ru; normalize_sms при опции возвращает текст без этих токенов.
**Вызовы в этом объекте:** `Path, ValueError, frozenset, path.read_text, path.read_text(encoding='utf-8').splitlines`.
**Ключевой смысл:** Список читается из локального resources; неизвестный язык отвергается. На текущем pipeline remove_stopwords=False, эффект качества ablation не проверен.

```text
34: raise ValueError("Допустимы только en/ru")
36: return frozenset(path.read_text(encoding="utf-8").splitlines())
```

**Быстрый поиск:** Ctrl+P → `ml/preprocessing.py`; Ctrl+G → `32`; Ctrl+F → `load_stopwords`.
**Показ преподавателю:** Открыть resources/stopwords_en.txt; показать флаг, не утверждать, что эта опция улучшила модель.

## 6. TF-IDF

**Путь и строки:** [ml/features.py:72](../../ml/features.py#L72), диапазон 72–103. **Объект:** `build_feature_pipeline`.
**Вход (сигнатура):** `*, include_char=True, remove_stopwords=False`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** sklearn.pipeline.Pipeline с FeatureUnion.
**Вызовы в этом объекте:** `FeatureUnion, ManualSmsFeatures, Pipeline, SmsNormalizer, TfidfVectorizer, transformers.append`.
**Ключевой смысл:** Два TfidfVectorizer учатся словарю/IDF только при fit; возврат конструктора ещё не означает обучение. Word параметры min_df2/max_df.95/sublinearTrue.

```text
79: TfidfVectorizer(
97: ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
103: return Pipeline([("features", FeatureUnion(transformers))])
```

**Быстрый поиск:** Ctrl+P → `ml/features.py`; Ctrl+G → `72`; Ctrl+F → `build_feature_pipeline`.
**Показ преподавателю:** Показать конструктор и отличить его от fit_transform в prepare_features.py.

## 7. Word N-grams

**Путь и строки:** [ml/features.py:72](../../ml/features.py#L72), диапазон 72–103. **Объект:** `build_feature_pipeline`.
**Вход (сигнатура):** `*, include_char=True, remove_stopwords=False`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** word-ветвь внутри Pipeline; после fit sparse n×8785 в данном эксперименте.
**Вызовы в этом объекте:** `FeatureUnion, ManualSmsFeatures, Pipeline, SmsNormalizer, TfidfVectorizer, transformers.append`.
**Ключевой смысл:** ngram_range=(1,2) создаёт слова и соседние пары; token_pattern сохраняет специальные токены. SmsNormalizer находится перед vectorizer.

```text
79: TfidfVectorizer(
97: ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
103: return Pipeline([("features", FeatureUnion(transformers))])
```

**Быстрый поиск:** Ctrl+P → `ml/features.py`; Ctrl+G → `72`; Ctrl+F → `build_feature_pipeline`.
**Показ преподавателю:** Вывести на бумаге униграммы/биграммы claim your prize, показать соответствующую строку ngram_range.

## 8. Character N-grams

**Путь и строки:** [ml/features.py:72](../../ml/features.py#L72), диапазон 72–103. **Объект:** `build_feature_pipeline`.
**Вход (сигнатура):** `*, include_char=True, remove_stopwords=False`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** char-ветвь с analyzer=char_wb; после fit n×51133.
**Вызовы в этом объекте:** `FeatureUnion, ManualSmsFeatures, Pipeline, SmsNormalizer, TfidfVectorizer, transformers.append`.
**Ключевой смысл:** include_char управляет наличием ветви, range3–5 задаёт длины; sublinear_tf здесь не задан и остаётся False.

```text
79: TfidfVectorizer(
97: ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
103: return Pipeline([("features", FeatureUnion(transformers))])
```

**Быстрый поиск:** Ctrl+P → `ml/features.py`; Ctrl+G → `72`; Ctrl+F → `build_feature_pipeline`.
**Показ преподавателю:** Найти char_wb, показать n-граммы слова bank с padding; не путать с char по всему предложению.

## 9. Ручные SMS-признаки

**Путь и строки:** [ml/features.py:49](../../ml/features.py#L49), диапазон 49–66. **Объект:** `ManualSmsFeatures.transform`.
**Вход (сигнатура):** `self, X`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** scipy.sparse.csr_matrix размером n×6.
**Вызовы в этом объекте:** `character.isalpha, character.isupper, check_is_fitted, int, len, matched_groups, normalize_sms, np.asarray, np.asarray(rows, dtype=np.float64).reshape, rows.append, sparse.csr_matrix, sum, text.count`.
**Ключевой смысл:** Исходный текст нужен для uppercase_fraction; cleaned нужен для <url>/<money>. len(matched_groups) читает frozen ml/rules.py, не расширенный runtime словарь.

```text
66: return sparse.csr_matrix(np.asarray(rows, dtype=np.float64).reshape(-1, len(MANUAL_NAMES)))
```

**Быстрый поиск:** Ctrl+P → `ml/features.py`; Ctrl+G → `49`; Ctrl+F → `transform`.
**Показ преподавателю:** Для PAY NOW! посчитать length8, uppercase1, count!1 и показать массив rows.

## 10. URL-признаки

**Путь и строки:** [ml/url_features.py:136](../../ml/url_features.py#L136), диапазон 136–157. **Объект:** `extract_url_features`.
**Вход (сигнатура):** `value: str`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict22 числовых признаков в фиксированном порядке.
**Вызовы в этом объекте:** `Levenshtein.distance, PSL, _model_feature_contract_v1, any, brand.split, host.endswith, host.split, int, label.encode, label.encode('ascii').decode, len, max, urlsplit, urlsplit(validate_url(value)).hostname.lower, validate_url`.
**Ключевой смысл:** Базовый v1 извлекает строки/числа без сети, wrapper уточняет reserved .test/.example и IDNA brand label. RF вызывает model_features, а не этот wrapper; входы не всегда равны таблице.

```text
157: return features
```

**Быстрый поиск:** Ctrl+P → `ml/url_features.py`; Ctrl+G → `136`; Ctrl+F → `extract_url_features`.
**Показ преподавателю:** Показать FEATURE_NAMES, validate_url, PSL и ветвь reserved; объяснить goggle.test: displaybrand1/modelbrand0.

## 11. Train/validation/test

**Путь и строки:** [ml/data/split_sms.py:7](../../ml/data/split_sms.py#L7), диапазон 7–29. **Объект:** `split_sms`.
**Вход (сигнатура):** `frame: pd.DataFrame, random_state: int=42`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict позиций train, val, test.
**Вызовы в этом объекте:** `ValueError, frame.text_clean.isna, frame.text_clean.isna().any, frame.text_clean.str.strip, frame.text_clean.str.strip().eq, frame.text_clean.str.strip().eq('').any, len, list, range, set, sorted, train_test_split`.
**Ключевой смысл:** Первый stratified split отделяет 30%, второй делит остаток пополам. Возвращаются сортированные позиции, а manifest дополнительно хранит source_ids/hash в prepare_features. URL split отдельно run по registrable_domain.

```text
9: raise ValueError("Нужен очищенный UCI с id/text_clean/label")
11: raise ValueError("Удалите дубликаты до разбиения")
13: raise ValueError("Пустой текст до разбиения")
15: raise ValueError("Нужны оба класса 0/1")
```

**Быстрый поиск:** Ctrl+P → `ml/data/split_sms.py`; Ctrl+G → `7`; Ctrl+F → `split_sms`.
**Показ преподавателю:** Показать два train_test_split и сохранённый manifest; проверить отсутствие общих групп по существующему тесту, не генерировать новый split.

## 12. Обучение Naive Bayes

**Путь и строки:** [ml/sms_experiment.py:209](../../ml/sms_experiment.py#L209), диапазон 209–264. **Объект:** `fit_candidates`.
**Вход (сигнатура):** `root: Path, grid=None`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** (rows, models, predictions) для всех train/validation кандидатов.
**Вызовы в этом объекте:** `(grid or GRID).items, (matrix.data < 0).any, (root / 'docs/reports/06_features.json').read_text, (root / f'data/processed/X_{name}.npz').read_bytes, ValueError, array_hash, choose_threshold, copy.deepcopy, data.text_raw.tolist, feature_model.transform, joblib.load, json.loads, len, load_partition, make_model, model.fit, model.named_steps['classifier'].fit, model.named_steps['scale'].fit_transform, model.predict_proba, np.abs, np.isfinite, np.isfinite(matrix.data).all, np.max, print, rows.append, sha256, sparse.load_npz, threadpool_limits, time.perf_counter, train.label.to_numpy, train.text_raw.tolist, val.label.to_numpy, val.text_raw.tolist, warnings.catch_warnings, warnings.simplefilter`.
**Ключевой смысл:** make_model создаёт MultinomialNB(alpha=value); fit_candidates обучает classifier.fit(scaled, train.label). Для NB scaled=X_train; проверяются nonnegative/finite/sparse.

```text
212: feature_model = joblib.load(root / "ml/artifacts/sms_feature_pipeline.joblib")
221: raise ValueError("Матрица признаков изменена")
227: raise ValueError("Нужна конечная неотрицательная sparse-матрица")
230: raise ValueError("Признаки не соответствуют исходному pipeline")
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `209`; Ctrl+F → `fit_candidates`.
**Показ преподавателю:** Найти algorithm==MultinomialNB и classifier.fit; показать baseline metadata без повторного обучения.

## 13. Обучение Logistic Regression

**Путь и строки:** [ml/sms_experiment.py:124](../../ml/sms_experiment.py#L124), диапазон 124–169. **Объект:** `make_model`.
**Вход (сигнатура):** `algorithm: str, value: float`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** необученный Pipeline; фактический fit в fit_candidates.
**Вызовы в этом объекте:** `CalibratedClassifierCV, LinearSVC, LogisticRegression, MaxAbsScaler, MultinomialNB, Pipeline, StratifiedKFold, ValueError, build_feature_pipeline`.
**Ключевой смысл:** LRbalanced, max_iter1000, liblinear, C=value; MaxAbsScaler fit только X_train. URLLR находится в url_experiment.run с StandardScaler и dense22 features.

```text
127: return Pipeline(features + [("classifier", MultinomialNB(alpha=value))])
129: return Pipeline(
162: return CalibratedClassifierCV(
169: raise ValueError("Неизвестный алгоритм")
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `124`; Ctrl+F → `make_model`.
**Показ преподавателю:** Показать обе конструкции LR и разные scaler; объяснить C и class_weight.

## 14. Обучение Linear SVM

**Путь и строки:** [ml/sms_experiment.py:124](../../ml/sms_experiment.py#L124), диапазон 124–169. **Объект:** `make_model`.
**Вход (сигнатура):** `algorithm: str, value: float`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** CalibratedClassifierCV с внутренним Pipeline.
**Вызовы в этом объекте:** `CalibratedClassifierCV, LinearSVC, LogisticRegression, MaxAbsScaler, MultinomialNB, Pipeline, StratifiedKFold, ValueError, build_feature_pipeline`.
**Ключевой смысл:** Внутри LinearSVCbalanced/dualFalse/max_iter10000; внешний CVsigmoid5 fold, ensembleFalse. Фактический model.fit(rawtrain, ytrain) в fit_candidates, чтобы и features обучались внутри каждого fold.

```text
127: return Pipeline(features + [("classifier", MultinomialNB(alpha=value))])
129: return Pipeline(
162: return CalibratedClassifierCV(
169: raise ValueError("Неизвестный алгоритм")
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `124`; Ctrl+F → `make_model`.
**Показ преподавателю:** Показать nested estimator и rawtrain fit; объяснить, почему нельзя заранее fit vocabulary на calibrationfold.

## 15. Обучение Random Forest

**Путь и строки:** [ml/url_experiment.py:22](../../ml/url_experiment.py#L22), диапазон 22–160. **Объект:** `run`.
**Вход (сигнатура):** `root: Path=ROOT`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** выбранный RF сохраняется joblib, результат эксперимента dict.
**Вызовы в этом объекте:** `(root / 'docs/reports/url_test_claim.json').exists, (root / 'docs/reports/url_test_claim.json').open, GroupShuffleSplit, GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42).split, LogisticRegression, RandomForestClassifier, RuntimeError, StandardScaler, archive.read_bytes, choose_threshold, features.groupby, features.groupby('label')[FEATURE_NAMES[:6]].mean, features.groupby('label')[FEATURE_NAMES[:6]].mean().to_dict, features.to_parquet, float, frame.canonical.isin, frame.canonical.map, frame.drop_duplicates, frame.dropna, frame.dropna().copy, frame.groupby, frame.groupby('canonical').label.nunique, frame.label.astype, frame.label.to_numpy, frame.reset_index, groups.tolist, int, joblib.dump, json.dump, json.dumps, len, make_pipeline, max, metrics, min, model.fit, model.predict_proba, model_features, model_url, n.endswith, next, normalized.append, now, np.asarray, np.quantile, parts.items, pd.DataFrame, pd.read_csv, print, report_path.exists, rows.append, set, sha256, splitter.split, time.perf_counter, train_test_split, v.tolist, write_json, y[v].sum, zipfile.ZipFile, zipped.namelist, zipped.open`.
**Ключевой смысл:** configs содержит RF150 деревьев, max_depth12/24, balanced, n_jobs2, seed42; model.fit(X[train], y[train]) не использует test.

```text
25: raise RuntimeError("URL test уже оценён: повторный подбор запрещён")
111: model.fit(X[train], y[train])
112: p = model.predict_proba(X[val])[:, 1]
122: pval = model.predict_proba(X[val])[:, 1]
```

**Быстрый поиск:** Ctrl+P → `ml/url_experiment.py`; Ctrl+G → `22`; Ctrl+F → `run`.
**Показ преподавателю:** Ctrl+F RandomForestClassifier и model.fit; открыть сохранённый selected=RF_depth12.

## 16. Подбор гиперпараметров

**Путь и строки:** [ml/sms_experiment.py:209](../../ml/sms_experiment.py#L209), диапазон 209–264. **Объект:** `fit_candidates`.
**Вход (сигнатура):** `root: Path, grid=None`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** таблица метрик, fitted модели и validation probabilities.
**Вызовы в этом объекте:** `(grid or GRID).items, (matrix.data < 0).any, (root / 'docs/reports/06_features.json').read_text, (root / f'data/processed/X_{name}.npz').read_bytes, ValueError, array_hash, choose_threshold, copy.deepcopy, data.text_raw.tolist, feature_model.transform, joblib.load, json.loads, len, load_partition, make_model, model.fit, model.named_steps['classifier'].fit, model.named_steps['scale'].fit_transform, model.predict_proba, np.abs, np.isfinite, np.isfinite(matrix.data).all, np.max, print, rows.append, sha256, sparse.load_npz, threadpool_limits, time.perf_counter, train.label.to_numpy, train.text_raw.tolist, val.label.to_numpy, val.text_raw.tolist, warnings.catch_warnings, warnings.simplefilter`.
**Ключевой смысл:** GRID3×3; цикл по algorithm/value учит train, проверяет val. row_key выбирает F2/Recall/Precision. Это ручная маленькая сетка, не GridSearchCV. URLconfigs аналогично 4 кандидата.

```text
212: feature_model = joblib.load(root / "ml/artifacts/sms_feature_pipeline.joblib")
221: raise ValueError("Матрица признаков изменена")
227: raise ValueError("Нужна конечная неотрицательная sparse-матрица")
230: raise ValueError("Признаки не соответствуют исходному pipeline")
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `209`; Ctrl+F → `fit_candidates`.
**Показ преподавателю:** Сопоставить GRID, цикл, select; объяснить ограничение аудита и не запускать select после testclaim.

## 17. Калибровка вероятностей

**Путь и строки:** [ml/sms_experiment.py:124](../../ml/sms_experiment.py#L124), диапазон 124–169. **Объект:** `make_model`.
**Вход (сигнатура):** `algorithm: str, value: float`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** обёртка CalibratedClassifierCV, обучаемая в fit_candidates.
**Вызовы в этом объекте:** `CalibratedClassifierCV, LinearSVC, LogisticRegression, MaxAbsScaler, MultinomialNB, Pipeline, StratifiedKFold, ValueError, build_feature_pipeline`.
**Ключевой смысл:** Sigmoid и StratifiedKFold5 shuffle seed42; ensemble=False. Порог выбирается позже отдельной choose_threshold, калибровка и threshold не одно и то же.

```text
127: return Pipeline(features + [("classifier", MultinomialNB(alpha=value))])
129: return Pipeline(
162: return CalibratedClassifierCV(
169: raise ValueError("Неизвестный алгоритм")
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `124`; Ctrl+F → `make_model`.
**Показ преподавателю:** Найти method=sigmoid и cv; нарисовать out-of-fold margin→P.

## 18. Расчёт ML-метрик

**Путь и строки:** [ml/sms_experiment.py:63](../../ml/sms_experiment.py#L63), диапазон 63–100. **Объект:** `metrics`.
**Вход (сигнатура):** `y, probabilities, threshold: float`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict Accuracy, P, R, F1, F2, ROC, PR, AP, macro, by_class, CM, threshold.
**Вызовы в этом объекте:** `((probabilities < 0) | (probabilities > 1)).any, (probabilities >= threshold).astype, ValueError, accuracy_score, auc, average_precision_score, confusion_matrix, confusion_matrix(y, predicted, labels=[0, 1]).tolist, enumerate, f1.mean, fbeta_score, float, int, np.asarray, np.isfinite, np.isfinite(probabilities).all, precision.mean, precision_recall_curve, precision_recall_fscore_support, recall.mean, roc_auc_score, set`.
**Ключевой смысл:** Входы проверяются на оба класса, форму и конечные P; p>=threshold превращается в predicted. auc(PR) трапецеидальный, average_precision_score отдельный.

```text
67: raise ValueError("Нужны оба класса и конечные вероятности")
69: raise ValueError("Некорректные вероятности")
75: return {
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `63`; Ctrl+F → `metrics`.
**Показ преподавателю:** Открыть функцию и вручную получить 88/94 и 88/93 из SMSCM; не пересчитывать held-out модель.

## 19. Сохранение моделей

**Путь и строки:** [ml/sms_experiment.py:272](../../ml/sms_experiment.py#L272), диапазон 272–346. **Объект:** `select`.
**Вход (сигнатура):** `root: Path=ROOT`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict selection; записываются SMSmodel, baseline и metadata.
**Вызовы в этом объекте:** `(artifacts / 'sms_baseline.joblib').read_bytes, (artifacts / 'sms_model.joblib').read_bytes, (reports / '07_test_claim.json').exists, (root / 'docs/datasets/uci_source.json').read_text, AssertionError, RuntimeError, array_hash, fit_candidates, input_fingerprint, joblib.dump, json.loads, load_partition, load_partition(root, 'val').label.to_numpy, max, metrics, now, np.save, platform.python_version, sha256, write_json`.
**Ключевой смысл:** joblib.dump фиксирует выбранный estimator до finaltest; metadata включает hash, threshold, source, versions. finalize дописывает testmetrics, ноне refit. URLdump в run.

```text
275: raise RuntimeError("Test уже зарезервирован/оценён; новый подбор запрещён")
281: joblib.dump(models[key], artifacts / "sms_model.joblib")
284: joblib.dump(models[baseline_key], artifacts / "sms_baseline.joblib")
302: raise AssertionError("Входные артефакты изменились во время обучения")
```

**Быстрый поиск:** Ctrl+P → `ml/sms_experiment.py`; Ctrl+G → `272`; Ctrl+F → `select`.
**Показ преподавателю:** Показать две dump-строки и metadata; объяснить, какие файлы вне Git и почему не принимаем чужой joblib.

## 20. Загрузка моделей

**Путь и строки:** [ml/inference.py:16](../../ml/inference.py#L16), диапазон 16–31. **Объект:** `ModelStore.__init__`.
**Вход (сигнатура):** `self, directory: Path`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** Объект ModelStore с sms/url и metadata; отсутствующая модель остаётся None.
**Вызовы в этом объекте:** `Path, ValueError, joblib.load, json.loads, meta.exists, meta.read_text, metadata.get, model.exists, model.read_bytes, setattr, sha256`.
**Ключевой смысл:** Sidecar читается до joblib; SHA сверяется перед загрузкой. Lifespan создаёт один store на процесс приложения; lru_cache обслуживает отдельный standalone helper.

```text
29: raise ValueError("Контрольная сумма модели не совпадает")
30: setattr(self, kind, joblib.load(model))
```

**Быстрый поиск:** Ctrl+P → `ml/inference.py`; Ctrl+G → `16`; Ctrl+F → `__init__`.
**Показ преподавателю:** Показать порядок hash→load и app.main.lifespan; рабочие артефакты не переименовывать.

## 21. SMS inference

**Путь и строки:** [ml/inference.py:33](../../ml/inference.py#L33), диапазон 33–41. **Объект:** `ModelStore.predict_sms`.
**Вход (сигнатура):** `self, text: str, language='en'`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict с label:int и probability:float.
**Вызовы в этом объекте:** `RuntimeError, ValueError, float, int, isinstance, len, re.search, self.sms.predict_proba, text.strip`.
**Ключевой смысл:** Вход проверяется на тип, длину и русский язык. predict_proba([text])[0,1] выбирает positiveP, label сравнивается с metadata.threshold. Пользовательский ввод не вызывает fit.

```text
35: raise ValueError("Текст должен содержать от 1 до 1000 символов")
37: raise ValueError("Русский ML не поддерживается; используйте правила и URL-анализ")
39: raise RuntimeError("SMS-модель недоступна")
40: probability = float(self.sms.predict_proba([text])[0, 1])
```

**Быстрый поиск:** Ctrl+P → `ml/inference.py`; Ctrl+G → `33`; Ctrl+F → `predict_sms`.
**Показ преподавателю:** Объяснить индексы[0,1], порог и отсутствие P_sms для ru в общем сервисе.

## 22. URL inference

**Путь и строки:** [ml/inference.py:43](../../ml/inference.py#L43), диапазон 43–46. **Объект:** `ModelStore.predict_url`.
**Вход (сигнатура):** `self, url: str`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** float, оценка P(phishing).
**Вызовы в этом объекте:** `RuntimeError, float, model_features, np.asarray, self.url.predict_proba`.
**Ключевой смысл:** model_features создаёт 22 координаты; np.asarray оборачивает их в матрицу одной строки. RF.predict_proba выбирает класс 1. model_url и frozen v1 находятся в url_features.py.

```text
45: raise RuntimeError("URL-модель недоступна")
46: return float(self.url.predict_proba(np.asarray([model_features(url)]))[0, 1])
```

**Быстрый поиск:** Ctrl+P → `ml/inference.py`; Ctrl+G → `43`; Ctrl+F → `predict_url`.
**Показ преподавателю:** Проследить путь строки до RF; никакого открытия адреса не требуется.

## 23. Подозрительные выражения

**Путь и строки:** [app/analysis/social.py:146](../../app/analysis/social.py#L146), диапазон 146–148. **Объект:** `matched_groups`.
**Вход (сигнатура):** `text`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** list[str], уникальные названия групп.
**Вызовы в этом объекте:** `PATTERNS.items, pattern.search, unicodedata.normalize, unicodedata.normalize('NFKC', text).lower`.
**Ключевой смысл:** PHRASES и PATTERNS строятся при импорте. Поиск использует NFKC/lower, re.escape и границы слова. Это runtime-словарь, отдельный от frozen ml/rules.py.

```text
148: return [name for name, pattern in PATTERNS.items() if pattern.search(text)]
```

**Быстрый поиск:** Ctrl+P → `app/analysis/social.py`; Ctrl+G → `146`; Ctrl+F → `matched_groups`.
**Показ преподавателю:** Показать CODE_REQUEST и comprehension; объяснить, почему k — число групп, не повторов фразы.

## 24. Risk Score

**Путь и строки:** [ml/risk.py:12](../../ml/risk.py#L12), диапазон 12–72. **Объект:** `compute_risk`.
**Вход (сигнатура):** `p_sms, p_url_max, se_groups, url_flags, *, t2=0.7, config_path=ROOT / 'risk_config.yaml', configuration=None`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict со score, level, category, explanation, recommendation, disclaimer.
**Вызовы в этом объекте:** `Path, Path(config_path).read_text, ValueError, _risk_result, any, contributions.get, f.get, len, max, min, reasons.append, reasons.insert, reasons.sort, round, titles.get, yaml.safe_load`.
**Ключевой смысл:** Читаются weights/levels/overrides. None исключает компоненту, ноль участвует. Sse=min(k/3,1) всегда имеет значение; после формулы overrides повышают минимум.

```text
28: raise ValueError("Вероятность вне диапазона 0–1")
72: return _risk_result(score, reasons, config)
```

**Быстрый поиск:** Ctrl+P → `ml/risk.py`; Ctrl+G → `12`; Ctrl+F → `compute_risk`.
**Показ преподавателю:** Посчитать пример из 02 и показать max для запроса кода.

## 25. Объяснения

**Путь и строки:** [ml/risk.py:75](../../ml/risk.py#L75), диапазон 75–103. **Объект:** `_risk_result`.
**Вход (сигнатура):** `score, reasons, config`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict итогового результата риска.
**Вызовы в этом объекте:** `max, min`.
**Ключевой смысл:** compute_risk собирает причины и ранжирует веса; _risk_result обрезает список до 6, добавляет fallback, рекомендацию и disclaimer. explain_url отдельно описывает flags; не все попадают в общий список.

```text
96: return {
```

**Быстрый поиск:** Ctrl+P → `ml/risk.py`; Ctrl+G → `75`; Ctrl+F → `_risk_result`.
**Показ преподавателю:** Показать fallback и[:6], объяснить известный пробел полноты, не называть причины атрибуцией RF.

## 26. API endpoints

**Путь и строки:** [app/web.py:280](../../app/web.py#L280), диапазон 280–281. **Объект:** `check_sms`.
**Вход (сигнатура):** `request: Request, user=Depends(current_user), session=Depends(db)`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** JSON результата perform после сериализации словаря FastAPI.
**Вызовы в этом объекте:** `Depends, perform, router.post`.
**Ключевой смысл:** Декоратор задаёт POST-маршрут, Depends проверяет CSRF/пользователя и предоставляет DB-сессию. check_url работает аналогично; история и stats находятся в том же модуле.

```text
281: return await perform(request, "SMS", user, session)
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `280`; Ctrl+F → `check_sms`.
**Показ преподавателю:** Найти @router и таблицу README; показать статус ошибки без изменения рабочей БД.

## 27. Регистрация пользователей

**Путь и строки:** [app/web.py:106](../../app/web.py#L106), диапазон 106–131. **Объект:** `register`.
**Вход (сигнатура):** `request: Request, session=Depends(db)`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict id/username при 201; HTTPException при плохом вводе или конфликте.
**Вызовы в этом объекте:** `Depends, User, UserProfile, data, fail, hash_password, isinstance, len, re.fullmatch, re.search, router.post, session.add, session.commit, session.rollback, str, str(values.get('email', '')).strip, str(values.get('email', '')).strip().lower, str(values.get('username', '')).strip, values.get`.
**Ключевой смысл:** Проверяются login/email/password, создаются солёный hash и UserProfile. Unique constraints страхуют гонку; IntegrityError приводит к rollback и понятной ошибке.

```text
127: session.commit()
131: return {"id": user.id, "username": user.username}
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `106`; Ctrl+F → `register`.
**Показ преподавателю:** Показать regex и except; объяснить, почему предварительного поиска дубликата недостаточно.

## 28. Авторизация и сессии

**Путь и строки:** [app/web.py:135](../../app/web.py#L135), диапазон 135–175. **Объект:** `login`.
**Вход (сигнатура):** `request: Request, session=Depends(db)`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** JSONResponse с session cookie; current_user отдельно возвращает User.
**Вызовы в этом объекте:** `AuthSession, Depends, JSONResponse, data, datetime.now, delete, delete(AuthSession).where, fail, failures.get, failures.pop, isinstance, len, new_token, request.cookies.get, response.set_cookie, router.post, select, select(User).where, session.add, session.commit, session.execute, session.scalar, str, time.monotonic, timedelta, token_hash, username.lower, values.get, verify_password`.
**Ключевой смысл:** Пауза и проверка hash предшествуют новой сессии. Token случайный, AuthSession хранит hash и expiry12 часов. Logout удаляет сессию в БД; смена пароля отзывает все.

```text
144: user = session.scalar(select(User).where(User.username == username))
156: session.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash(old)))
165: session.commit()
175: return response
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `135`; Ctrl+F → `login`.
**Показ преподавателю:** Показать set_cookie/current_user/logout, не выводить реальные токены.

## 29. Хеширование паролей

**Путь и строки:** [app/core/security.py:10](../../app/core/security.py#L10), диапазон 10–13. **Объект:** `hash_password`.
**Вход (сигнатура):** `password: str`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** str algorithm$iterations$salt$digest; verify_password возвращает bool.
**Вызовы в этом объекте:** `bytes.fromhex, hashlib.pbkdf2_hmac, hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), ITERATIONS).hex, password.encode, secrets.token_hex`.
**Ключевой смысл:** Соль 16 байт и PBKDF2-HMAC-SHA256600000; compare_digest проверяет digest. token_hash использует быстрый SHA256 для высокоэнтропийного случайного token, не для пароля.

```text
13: return f"pbkdf2_sha256${ITERATIONS}${salt}${digest}"
```

**Быстрый поиск:** Ctrl+P → `app/core/security.py`; Ctrl+G → `10`; Ctrl+F → `hash_password`.
**Показ преподавателю:** Различить хеш и шифрование, объяснить роль соли; реальные hashes пользователей не показывать.

## 30. CSRF

**Путь и строки:** [app/web.py:64](../../app/web.py#L64), диапазон 64–68. **Объект:** `csrf`.
**Вход (сигнатура):** `request: Request`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** None при успехе, HTTP403 при ошибке.
**Вызовы в этом объекте:** `fail, hmac.compare_digest, request.cookies.get, request.headers.get`.
**Ключевой смысл:** Сравниваются sg_csrf cookie и x-csrf-token header. Шаблон page выводит meta, api() переносит token в заголовок. Это double-submit, а не обычная проверка пароля.

```text
64: def csrf(request: Request):
65: expected = request.cookies.get("sg_csrf", "")
66: actual = request.headers.get("x-csrf-token", "")
67: if not expected or not actual or not hmac.compare_digest(expected, actual):
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `64`; Ctrl+F → `csrf`.
**Показ преподавателю:** Открыть серверный csrf и браузерный api рядом, найти тест 403 без токена.

## 31. SQLAlchemy

**Путь и строки:** [app/db/session.py:27](../../app/db/session.py#L27), диапазон 27–28. **Объект:** `build_session_factory`.
**Вход (сигнатура):** `engine`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** sessionmaker; build_engine отдельно возвращает Engine.
**Вызовы в этом объекте:** `sessionmaker`.
**Ключевой смысл:** Base описывает общую declarative metadata. Engine включает SQLite/FK/StaticPool для соответствующего режима. Фабрика даёт SQLAlchemy Session — единицу работы, не строку AuthSession.

```text
28: return sessionmaker(bind=engine, expire_on_commit=False)
```

**Быстрый поиск:** Ctrl+P → `app/db/session.py`; Ctrl+G → `27`; Ctrl+F → `build_session_factory`.
**Показ преподавателю:** Показать Base, build_engine, Check; объяснить select/where/commit.

## 32. Миграции Alembic

**Путь и строки:** [migrations/versions/b2a3401dc933_users_sessions_checks_and_model_versions.py:12](../../migrations/versions/b2a3401dc933_users_sessions_checks_and_model_versions.py#L12), диапазон 12–105. **Объект:** `upgrade`.
**Вход (сигнатура):** `()` — нет явных аргументов. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** None; функция выполняет DDL создания таблиц, индексов и ограничений.
**Вызовы в этом объекте:** `op.create_index, op.create_table, op.f, sa.Boolean, sa.Column, sa.DateTime, sa.Float, sa.ForeignKeyConstraint, sa.Integer, sa.JSON, sa.PrimaryKeyConstraint, sa.String, sa.Text, sa.UniqueConstraint`.
**Ключевой смысл:** migrations/env.py передаёт Base.metadata, upgrade задаёт явную схему, app/__main__.py вызывает upgrade head. Downgrade описывает обратное удаление.

```text
12: def upgrade():
13: # ### commands auto generated by Alembic - please adjust! ###
14: op.create_table(
15: "model_versions",
```

**Быстрый поиск:** Ctrl+P → `migrations/versions/b2a3401dc933_users_sessions_checks_and_model_versions.py`; Ctrl+G → `12`; Ctrl+F → `upgrade`.
**Показ преподавателю:** Показать upgrade и тест fresh schema; downgrade рабочей базы не выполнять.

## 33. Сохранение истории

**Путь и строки:** [app/web.py:191](../../app/web.py#L191), диапазон 191–276. **Объект:** `perform`.
**Вход (сигнатура):** `request, kind, user, session`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** Сериализованный Check как dict, при этом запись сохранена с дочерними результатами.
**Вызовы в этом объекте:** `''.join, ''.join((c for c in text if c in '\n\t' or ord(c) >= 32)).strip, Check, DetectedFeature, UrlResult, analyze, copy.deepcopy, data, fail, isinstance, len, logging.getLogger, logging.getLogger('sms_guard').info, mask_text, ord, recent.append, recent.popleft, request.headers.get, select, select(Check).where, serialize, session.add, session.commit, session.flush, session.rollback, session.scalar, str, text.strip, time.monotonic, time.perf_counter, uuid4, values.get`.
**Ключевой смысл:** Порядок: idempotency→analyze→mask→Check/flush→children→commit. Без consent input_text=None; URL-параметры SMS тоже маскируются. Конфликт unique откатывается и возвращает существующую запись.

```text
206: select(Check).where(Check.user_id == user.id, Check.request_key == key)
209: return serialize(existing)
246: session.flush()
261: session.commit()
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `191`; Ctrl+F → `perform`.
**Показ преподавателю:** Показать границу транзакции и различить flush/commit/rollback.

## 34. Разграничение истории

**Путь и строки:** [app/web.py:311](../../app/web.py#L311), диапазон 311–315. **Объект:** `detail`.
**Вход (сигнатура):** `check_id: int, user=Depends(current_user), session=Depends(db)`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict своей проверки либо HTTP404.
**Вызовы в этом объекте:** `Depends, fail, router.get, select, select(Check).where, serialize, session.scalar`.
**Ключевой смысл:** Условие содержит одновременно Check.id и Check.user_id. history/remove/clear_history тоже ограничены владельцем. UI-фильтра без серверной проверки было бы недостаточно.

```text
312: check = session.scalar(select(Check).where(Check.id == check_id, Check.user_id == user.id))
315: return serialize(check)
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `311`; Ctrl+F → `detail`.
**Показ преподавателю:** Разобрать тест Alice/Bob и чужие GET/DELETE404.

## 35. Dashboard и SQL-агрегации

**Путь и строки:** [app/web.py:372](../../app/web.py#L372), диапазон 372–433. **Объект:** `stats`.
**Вход (сигнатура):** `request: Request, user=Depends(current_user), session=Depends(db)`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** dict агрегатов, timeline, top_features, recent, models.
**Вызовы в этом объекте:** `Check.created_at.desc, Check.id.desc, Depends, categories.values, conditions.append, datetime.now, dict, fail, func.avg, func.count, func.count().desc, func.date, int, request.query_params.get, router.get, select, select(Check).where, select(Check).where(*conditions).order_by, select(Check).where(*conditions).order_by(Check.created_at.desc(), Check.id.desc()).limit, select(Check.category, func.count()).where, select(Check.category, func.count()).where(*conditions).group_by, select(Check.check_type, func.count()).where, select(Check.check_type, func.count()).where(*conditions).group_by, select(Check.risk_level, func.count()).where, select(Check.risk_level, func.count()).where(*conditions).group_by, select(DetectedFeature.code, func.count()).join, select(DetectedFeature.code, func.count()).join(Check, DetectedFeature.check_id == Check.id).where, select(DetectedFeature.code, func.count()).join(Check, DetectedFeature.check_id == Check.id).where(*conditions).group_by, select(DetectedFeature.code, func.count()).join(Check, DetectedFeature.check_id == Check.id).where(*conditions).group_by(DetectedFeature.code).order_by, select(DetectedFeature.code, func.count()).join(Check, DetectedFeature.check_id == Check.id).where(*conditions).group_by(DetectedFeature.code).order_by(func.count().desc()).limit, select(ModelVersion).where, select(func.avg(Check.risk_score)).where, select(func.date(Check.created_at), Check.category, func.count()).where, select(func.date(Check.created_at), Check.category, func.count()).where(*conditions).group_by, select(func.date(Check.created_at), Check.category, func.count()).where(*conditions).group_by(func.date(Check.created_at), Check.category).order_by, serialize, session.execute, session.execute(select(Check.category, func.count()).where(*conditions).group_by(Check.category)).all, session.execute(select(Check.check_type, func.count()).where(*conditions).group_by(Check.check_type)).all, session.execute(select(Check.risk_level, func.count()).where(*conditions).group_by(Check.risk_level)).all, session.execute(select(DetectedFeature.code, func.count()).join(Check, DetectedFeature.check_id == Check.id).where(*conditions).group_by(DetectedFeature.code).order_by(func.count().desc()).limit(5)).all, session.execute(select(func.date(Check.created_at), Check.category, func.count()).where(*conditions).group_by(func.date(Check.created_at), Check.category).order_by(func.date(Check.created_at))).all, session.scalar, session.scalars, session.scalars(select(Check).where(*conditions).order_by(Check.created_at.desc(), Check.id.desc()).limit(10)).all, sum, timedelta`.
**Ключевой смысл:** SQL COUNT/GROUP BY/AVG/date и join считают сводку. Период 7/30/90; all разрешён staff, recent в этом режиме пуст, чтобы не выдавать чужие тексты.

```text
384: select(Check.category, func.count()).where(*conditions).group_by(Check.category)
389: select(Check.check_type, func.count()).where(*conditions).group_by(Check.check_type)
394: select(Check.risk_level, func.count()).where(*conditions).group_by(Check.risk_level)
399: .where(*conditions)
```

**Быстрый поиск:** Ctrl+P → `app/web.py`; Ctrl+G → `372`; Ctrl+F → `stats`.
**Показ преподавателю:** Показать group_by, условие staff и JS dashboard; API-время не называть временем браузера.

## 36. JavaScript-запросы

**Путь и строки:** [static/js/app.js:7](../../static/js/app.js#L7), диапазон 7–11. **Объект:** `api`.
**Вход (сигнатура):** `api(url, method="GET", body, extra={})`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** Promise с JSON либо null для 204; выбрасывает Error при неуспешном статусе.
**Вызовы в этом объекте:** `fetch, JSON.stringify, response.json, Error`.
**Ключевой смысл:** fetch посылает method, JSON body, credentials same-origin и CSRF-header из meta. Для 204 нет JSON-чтения, иначе response.json и проверка status.

```text
7: async function api(url,method='GET',body,extra={}){
8:  const response=await fetch(url,{method,credentials:'same-origin',headers:{'Content-Type':'application/json','X-CSRF-Token':$('meta[name=csrf-token]').content,...extra},body:body===undefined?undefined:JSON.stringify(body)});
9:  if(response.status===204)return null;
10:  const result=await response.json();if(!response.ok)throw Error(result.error?.message||'Не удалось выполнить запрос');return result;
```

**Быстрый поиск:** Ctrl+P → `static/js/app.js`; Ctrl+G → `7`; Ctrl+F → `api`.
**Показ преподавателю:** Показать запрос в Network без демонстрации секретных cookie, объяснить finally формы.

## 37. Отображение результатов

**Путь и строки:** [static/js/app.js:18](../../static/js/app.js#L18), диапазон 18–29. **Объект:** `renderResult`.
**Вход (сигнатура):** `renderResult(item)`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** Изменяет DOM блока результата, осмысленного возвращаемого значения нет.
**Вызовы в этом объекте:** `el, badge, Object.entries, api, confirm, Date`.
**Ключевой смысл:** el создаёт узлы и textContent. renderResult выводит badge, score, P, причины, неактивные URL, features, совет и disclaimer; удаление требует подтверждения.

```text
18: function renderResult(item){
19:  const box=$('#result');box.className='card p-4 shadow-sm mt-4';box.replaceChildren();box.append(badge(item.category),el('h2',`Риск: ${({LOW:'низкий',MEDIUM:'средний',HIGH:'высокий'})[item.risk_level]} · ${item.risk_score}/100`,'h4 mt-3'));
20:  const bar=el('progress');bar.max=100;bar.value=item.risk_score;bar.className='w-100 mb-3';box.append(bar);
21:  box.append(el('p',item.input||'','text-break'));
```

**Быстрый поиск:** Ctrl+P → `static/js/app.js`; Ctrl+G → `18`; Ctrl+F → `renderResult`.
**Показ преподавателю:** Показать el и отсутствие href у найденных адресов, экспериментальное предупреждение URL.

## 38. Unit и integration tests

**Путь и строки:** [tests/test_product.py:164](../../tests/test_product.py#L164), диапазон 164–175. **Объект:** `test_auth_csrf_logout_and_owner_protection`.
**Вход (сигнатура):** `product`. Аргументы `request/session/user` поступают от FastAPI Depends; `self` — экземпляр класса.
**Результат:** None; asserts подтверждают ожидаемые статусы и изоляцию.
**Вызовы в этом объекте:** `account, product.delete, product.get, product.get('/api/v1/checks/').json, product.post, product.post('/api/v1/check/sms/', json={'text': 'Hello, see you tomorrow'}).json`.
**Ключевой смысл:** Fixture product создаёт временную SQLite и TestClient. Alice создаёт запись, logout ведёт к 401, Bob не видит и не удаляет её 404. Отдельные unit-файлы проверяют normalize/features/EDA.

```text
164: def test_auth_csrf_logout_and_owner_protection(product):
165: assert product.get("/history/", follow_redirects=False).status_code == 303
166: assert product.post("/api/v1/auth/register/", json={}).status_code == 403
167: account(product)
```

**Быстрый поиск:** Ctrl+P → `tests/test_product.py`; Ctrl+G → `164`; Ctrl+F → `test_auth_csrf_logout_and_owner_protection`.
**Показ преподавателю:** Назвать предусловие, действие и ожидаемый результат каждого assert; не приписывать исторический 114passed новому запуску.

## Общая навигация по наиболее важным файлам

| Файл | Как читать | Что не перепутать |
|---|---|---|
| `ml/sms_experiment.py` | metrics63 → choose_threshold103 → make_model124 → fit_candidates209 → select272 → finalize377 | make_model не учит; fit_candidates учит; finalize только финально оценивает и защищён claim |
| `ml/url_experiment.py` | run22:read/clean → group split →features →configs/fit →threshold →dump/test | Загрузка ZIP сети здесь отсутствует; всё из локального архива |
| `ml/preprocessing.py` | regex ы → normalization_steps49 → normalize_sms83 →clean_sms92 | normalize чистая строковая функция; clean работаетс DataFrame |
| `ml/features.py` | SmsNormalizer26 →ManualSmsFeatures44 →build_feature_pipeline72 | WordsublinearTrue; chardefaultFalse; manual читает raw |
| `ml/url_features.py` | validate53 →v1 контракт 88 →publicwrapper136 →model_url160/model_features173 →explain178 | display/rulesfeatures невсегдаравны RFinputs |
| `ml/inference.py` | ModelStore15 →load16 →SMS33/URL43 →cachedhelper50 | Загрузка once, никакого fit прианализе |
| `ml/risk.py` | compute_risk12 →_risk_result75 →compute_url_risk106 | Score не probability; standaloneURLmapping особый |
| `app/analysis/services.py` | AnalysisResult14 →analyze_url35 →analyze59 →mask114 | Язык/правилаперед SMSpipeline; ru без P_sms |
| `app/analysis/social.py` | PHRASES →PATTERNS →matched_groups146 | Эторасширенный runtime словарь, не frozenml/rules |
| `app/web.py` | helpers35–102 →auth106–188 →perform191–276 →history290 →stats372 →page445 | Приложения accounts/dashboard пакеты-маркеры; реальные маршруты здесь |
| `app/main.py` | create_app20 →lifespan24 →exceptionhandlers92/102 | startup загружаетмодели, создаёт схему и обновляет версии |
| `app/db/models.py` | User26, Profile37, AuthSession46, Check53, URL85, Feature96, Version105 | DBSession и AuthSession—разныепонятия |
| `app/core/security.py` | hash10, verify16, new_token29, token_hash33 | PBKDF2 дляпароля; SHA дляслучайного токена |
| `static/js/app.js` | api7 →table13 →renderResult18 →chart31 →dashboard32 →init36 | Файлплотный; дополнительный поиск имён надёжнее одного Ctrl+G |

## Другие файлы и инструменты

`app/core/config.py:Settings` читает env; `logging.py:configure_logging` задаёт rotation; `app/db/session.py` создаёт Engine/Session/FK. `ml/eda.py` считает баланс/длины/topn/доли и сохраняет PNG; `scripts/build_eda_notebook.py` строит notebook, `run_notebook.py` исполняет и сохраняет outputs; `calibrate_risk.py` подбирает EN веса и **пишет risk_config.yaml**, поэтому при подготовке не запускался. `verify_mvp.py` использует временную БД/нагрузочные фикстуры; `browser_smoke.py` добавляет демо-проверки и перезаписывает скриншоты. `app/seed_demo.py` создаёт демоаккаунт и до 30 проверок, не предсказания-заглушки.

`templates/base.html` задаёт навигацию/meta/static, `templates/app.html` ветвится по path, `static/css/app.css` уточняет Bootstrap. `risk/__init__.py` — документирующий пакет, реальный расчёт в `ml/risk.py`; пустой пакет не доказывает реализации. Notebook01 — выполненный EDA;02 — сохранённые сравнения/кривые/ошибки/reload и повторный train-fit URL сравнения. Не исполнять его автоматически при показе.

## Значение входных параметров карты

- `root` — корень проекта с данными и reports; `directory` — папка доверенных моделей; `grid` — словарь алгоритм→список настроек, по умолчанию GRID. Это файловые пути/настройки эксперимента, не произвольный URL пользователя.
- `algorithm` — имя семейства, `value` — alpha для NB либо C для LR/SVM. `include_char` включает символьную ветвь; `remove_stopwords` включает фильтрацию; `language` выбирает поддержанный список/режим.
- `text` — одна строка, `X` — последовательность raw SMS у custom-transformer, а после vectorizer числовая матрица; `y` — истинные0/1, `probabilities` — оценки positive той же длины, `threshold` — граница решения, `min_precision` — ограничение выбора.
- `p_sms/p_url_max` — вероятность или None при отсутствии, `se_groups` — список групп, `url_flags` — список словарей признаков. `t2` — верхний порог URL, `configuration` — уже прочитанный dict, `config_path` — путь YAML. Ноль и None имеют разный смысл.
- `request` — HTTP-запрос, `user` — текущий авторизованный User, `session` — SQLAlchemy Session, `check_id` — ID записи, `kind` — SMS/URL. Вспомогательный `perform` получает уже проверенные зависимости, он не самостоятельный публичный маршрут.
- В JS `api(url,method,body,extra)` получает локальный API-путь, HTTP-метод, объект JSON и дополнительные headers; `renderResult(item)` получает JSON проверки, а не текст HTML. `records/target` у table — список записей и DOM-селектор.

`upgrade()` не принимает пользовательских параметров: контекст Alembic уже подключён env.py. Pytest fixture `product` — тестовый клиент, `tmp_path` — изолированная папка. Эти параметры нельзя подменять рабочей БД ради тренировки.
