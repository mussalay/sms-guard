"""Execute task-5 notebook with THIS Python, using a workspace-local kernelspec."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    from ml.data.load_sms import write_json

    data_dir = ROOT / ".cache/jupyter"
    kernel_dir = data_dir / "kernels/sms-guard"
    write_json(
        kernel_dir / "kernel.json",
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "SMS Guard (.venv)",
            "language": "python",
        },
    )
    os.environ["JUPYTER_PATH"] = str(data_dir)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(data_dir / "runtime")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    os.environ["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    notebook_path = ROOT / "notebooks/01_eda_sms.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    # NotebookClient's working directory is explicit; selected kernel uses sys.executable.
    NotebookClient(
        notebook, timeout=180, kernel_name="sms-guard", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    nbformat.write(notebook, notebook_path)
    print("Executed notebooks/01_eda_sms.ipynb; no cell errors")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
