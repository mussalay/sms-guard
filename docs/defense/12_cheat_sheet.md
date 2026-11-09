# 12. Шпаргалка перед защитой

## Числа и оговорки

- en — основной SMS ML. ru — **rules+URL, P_sms=null**. kk исключён DEV-01.
- UCI5574→5135; split **3594/770/771**, seed42.
- **59924=8785word+51133char+6manual**: число координат, не SMS.
- NB/LR/calibratedSVM; выбрана **SVM C10**, порог **.1135616733**, максимум validationF2 при P≥.85.
- SMS **P.936170, R.946237, F1.941176, F2.944206, ROC.996035, PR.985589**. Цели достигнуты.
- SMS CM: **TN672, FP6, FN5, TP88**.
- URL50000, split **33360/10121/6519**, домены не пересекаются, **22**признака.
- URL **RF150 деревьев, depth12**, t1=.1440726172, t2=.2881452343.
- URL **P.642469, R.743486, F1.689296, F2.720819, ROC.844709, PR.835703**. Цели не достигнуты.
- URL CM: **TN3372, FP921, FN571, TP1655**.
- Ru100 искусственных: Recall **.66**, это не SMS ML.
- Risk **.5/.3/.2**, S=min(k/3,1), отсутствие компонента исключает её вес; levels **22/70**, overrides **70/85/60**.
- Пароли: PBKDF2-SHA256 **600000** и salt; opaque sessions12 часов, CSRF, owner404.
- Аудит: **8 полных/10 частичных**,114tests,78%overall, ML≈71%, risk98%.

## Формулы

Precision=TP/(TP+FP), Recall=TP/(TP+FN), Accuracy=(TP+TN)/N.

F1=2PR/(P+R); F2=5PR/(4P+R). β2 означает β²4 в весах, а не F2=2F1.

WordTF-IDF: idf=ln((1+N)/(1+df))+1, tf=1+ln(count), затем L2. У char sublinear=False по умолчанию.

SVM margin f=w·x+b; sigmoid-калибратор учит отображение в P. Порог превращает P в label; RiskScore объединяет несколько сигналов. Это четыре разных понятия.

## Поиск за 10 секунд

|Что показать|Ctrl+P → Ctrl+F|
|---|---|
|SMSfit|`ml/sms_experiment.py` →`fit_candidates`|
|SVM/calibration|тот жефайл→`LinearSVC`/`CalibratedClassifierCV`|
|NLP|`ml/preprocessing.py`→`normalize_sms`|
|TF-IDF|`ml/features.py`→`TfidfVectorizer`|
|URLfit|`ml/url_experiment.py`→`model.fit`|
|Inference|`ml/inference.py`→`predict_sms`/`predict_url`|
|Риск|`ml/risk.py`→`compute_risk`|
|API|`app/web.py`→`@router.post`|
|БД/история|`app/db/models.py`→`class Check`;`app/web.py`→`history`|
|Пароли|`app/core/security.py`→`hash_password`|
|Dashboard|`app/web.py`→`stats`;`static/js/app.js`→`dashboard`|

Подробные проверенные строки — [03](03_code_map.md).

## Ответы на неудобные вопросы

«URL-модель реальная, но не достигла целей. Улучшать её нужно с независимой оценкой, не подбором по test». «Русский не обучался: здесь правила и URL». «Score — эвристика, не вероятность банковского мошенничества». «Работающий MVP не закрывает все критерии академической сдачи».

На защите не запускать обучение/finalize/calibrate_risk/run_notebook/seed ради эффекта. Не открывать подозрительные URL. SAFE не означает гарантию; новый ввод вычисляется моделью, а не ищется в таблице готовых ответов.

Запуск: `.\.venv\Scripts\python.exe -m app`; сайт http://127.0.0.1:8080/. Доверенные модели и metadata должны быть на месте. Примеры с фактическими категориями —10.
