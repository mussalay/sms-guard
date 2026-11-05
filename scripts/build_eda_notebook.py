"""Build the auditable task-5 notebook; execute separately with run_notebook.py."""

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "SMS Guard (.venv)",
        "language": "python",
        "name": "sms-guard",
    }
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook.cells = [
        md(
            "# Задание 5. EDA очищенных UCI SMS\n\n"
            "Исследуется весь очищенный английский корпус до обучения, как разрешает критерий ТЗ. "
            "Параметры/пороги по этим данным не подбираются. Русский test-only не читается; "
            "казахский исключён DEV-01. Spam не тождественен мошенничеству. "
            "Тексты сообщений не выводятся: только агрегаты и токены."
        ),
        code(
            "from pathlib import Path\nimport sys\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'ml').exists():\n    ROOT = ROOT.parent\n"
            "assert (ROOT / 'ml').exists(), 'Запустите из корня проекта или notebooks/'\n"
            "sys.path.insert(0, str(ROOT))\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n"
            "from ml.data.load_sms import sha256, write_json\n"
            "from ml.eda import summarize, plot_eda, observations\n"
            "import json\n"
            "source = ROOT / 'data/processed/sms_clean.csv'\n"
            "audit_path = ROOT / 'docs/reports/04_preprocessing.json'\n"
            "audit = json.loads(audit_path.read_text(encoding='utf-8'))\n"
            "assert sha256(source.read_bytes()) == audit['clean_sha256']\n"
            "sms = pd.read_csv(source, keep_default_na=False)\n"
            "assert sms.text_clean.is_unique and sms.id.is_unique\n"
            "assert not sms.text_clean.str.strip().eq('').any()\n"
            "features, summary = summarize(sms)\n"
            "print('Число строк:', len(sms))\nprint(sms.dtypes.to_string())"
        ),
        md(
            "## Баланс классов и длина\n\nДлины измеряются по исходному тексту "
            "сохранённых после очистки строк; "
            "для сравнения таблица содержит также длины нормализованного текста. "
            "Слова длины — split по пробелам. "
            "Boxplot показывает выбросы, строки не удаляются по длине."
        ),
        code(
            "class_table = pd.DataFrame(summary['classes']).T\n"
            "display(class_table[['count', 'percentage', 'mean_chars_raw', 'median_chars_raw', "
            "'mean_words_raw', 'mean_chars_clean', 'mean_words_clean']])\n"
            "figures = plot_eda(features, summary, ROOT / 'docs/figures')\n"
            "for path in figures[:2]:\n    display(Image(filename=str(path)))"
        ),
        md(
            "## Топ-20 слов и биграмм каждого класса\n\nNLTK RegexpTokenizer; локальные английские "
            "стоп-слова NLTK удаляются до формирования биграмм внутри каждого сообщения. "
            "Сущности остаются токенами. Биграммы не пересекают границы SMS. "
            "Считаются вхождения, а не число документов."
        ),
        code(
            "for label, name in [('0', 'ham'), ('1', 'spam')]:\n"
            "    print(name)\n"
            "    for key in ['top20_words', 'top20_bigrams']:\n"
            "        values = summary['classes'][label][key]\n"
            "        display(pd.DataFrame(values, columns=['Слово / биграмма', 'Число']))\n"
            "for path in figures[2:4]:\n    display(Image(filename=str(path)))"
        ),
        md(
            "## Доли URL, телефонов, сумм и !\n\n"
            "Доля сообщений с наличием хотя бы одного соответствующего токена/символа, "
            "отдельно по классам."
        ),
        code(
            "display(pd.DataFrame({label: values['indicator_percentages'] "
            "for label, values in summary['classes'].items()}))\n"
            "display(Image(filename=str(figures[4])))"
        ),
        md("## Наблюдения и ограничения\n\nВсе следующие числа рассчитаны выполненными ячейками."),
        code(
            "facts = observations(summary)\n"
            "display(Markdown(chr(10).join('- ' + fact for fact in facts)))\n"
            "summary['source_sha256'] = sha256(source.read_bytes())\n"
            "summary['observations'] = facts\n"
            "write_json(ROOT / 'docs/reports/05_eda.json', summary)\n"
            "report = '# Задание 5 — фактические результаты EDA' + chr(10)*2\n"
            "report += chr(10).join('- ' + f for f in facts)\n"
            "report += chr(10)*2\n"
            "report += 'Исследован весь очищенный UCI до обучения без подбора параметров. '\n"
            "report += 'Русский test-only не использован. Пять PNG — docs/figures/. '\n"
            "report += 'Дисбаланс учитывается стратификацией; для LR планируется balanced, '\n"
            "report += 'далее Precision/Recall/F1/F2, а не только Accuracy. '\n"
            "report += 'Модель не обучена, качество не измерено.' + chr(10)\n"
            "(ROOT / 'docs/05_eda.md').write_text(report, encoding='utf-8')"
        ),
        md(
            "Дисбаланс требует стратификации, class_weight для LR и оценки Precision/Recall/F1/F2. "
            "Решения об алгоритме не принимаются в этом notebook. "
            "Облако слов опционально и не строилось. "
            "Наблюдения корпуса не доказывают качество на новых SMS, "
            "особенно русском языке."
        ),
    ]
    # Keep regenerated cells compliant with the same formatter/import order as repository code.
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "I",
                "--fix",
                "--stdin-filename",
                "cell.py",
                "-",
            ],
            input=cell.source,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        formatted = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--stdin-filename", "cell.py", "-"],
            input=result.stdout,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        cell.source = formatted.stdout.strip()
    nbf.write(notebook, ROOT / "notebooks/01_eda_sms.ipynb")


if __name__ == "__main__":
    build()
