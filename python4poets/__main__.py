"""
Check that python4poets is set up and working:

    python -m python4poets

Prints one line per check, and what to do about anything that fails.
"""

import importlib.util
import os
import platform
import sys


def check():
    ok = True

    def report(passed, what, advice=""):
        nonlocal ok
        ok = ok and passed
        print(f"  {'OK  ' if passed else 'FAIL'}  {what}")
        if not passed and advice:
            print(f"        -> {advice}")

    print(f"python4poets setup check (Python {platform.python_version()}, {platform.system()})")
    print(f"  Python: {sys.executable}")

    major, minor = sys.version_info[:2]
    report((3, 10) <= (major, minor), f"Python {major}.{minor} is supported", "Install Python 3.13 from python.org.")
    if (major, minor) > (3, 13):
        print("  note  Some packages (e.g. gensim) don't yet support Python above 3.13.")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv and "google.colab" not in sys.modules:
        print("  note  Not in a virtual environment (fine in Colab; on your own computer, use .venv).")

    try:
        import python4poets as p4p

        report(True, "python4poets imports")
    except Exception as err:  # noqa: BLE001 - report anything
        report(False, f"python4poets imports ({err})", "Reinstall the package.")
        return False

    try:
        report(len(p4p.get_stopwords("english")) > 100, "stopword lists (no download needed)")
    except Exception as err:  # noqa: BLE001
        report(False, f"stopword lists ({err})", "Reinstall the package.")

    folder = os.path.join(p4p._corpus.DATA_FOLDER, "gutenberg")
    try:
        catalog = p4p.gutenberg_catalog(folder=folder)
        report(len(catalog) > 1000, f"downloads work: Gutenberg catalog, {len(catalog):,} books, in {os.path.abspath(folder)}")
    except Exception as err:  # noqa: BLE001
        report(False, f"downloads ({type(err).__name__}: {err})",
               "Check your internet connection. If an AI assistant ran this, it needs to run it with network access enabled.")

    try:
        import pandas as pd

        df = pd.DataFrame({"year": [2000, 2001, 2002], "txt": ["poetry poetry", "poetry prose", "prose prose"]})
        path = os.path.join(p4p._corpus.DATA_FOLDER, "setup_check.png")
        p4p.plot_word_over_time(df, "poetry").save(path, verbose=False)
        report(os.path.exists(path), f"plots: saved a test plot to {path}")
    except Exception as err:  # noqa: BLE001
        report(False, f"plots ({type(err).__name__}: {err})")

    for module, what in [("ipykernel", "Jupyter notebooks"), ("datasets", "Hugging Face datasets")]:
        if importlib.util.find_spec(module):
            print(f"  OK    {what} (optional) installed")
        else:
            print(f"  --    {what} (optional) not installed; to add it: pip install {'jupyter' if module == 'ipykernel' else module}")

    print("All set!" if ok else "Something needs fixing: see FAIL above.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
