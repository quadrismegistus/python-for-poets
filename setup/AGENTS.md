# Rules for AI assistants in this folder

This folder is for the course *Python for Poets*. The person you're working with is a humanities student who is learning Python. Follow these rules in everything you do here.

## Python and packages

- Use the Python in `.venv` for everything: `.venv/bin/python` on a Mac, `.venv\Scripts\python.exe` on Windows. Install packages with that Python's pip (`python -m pip install ...`), and ask the student before installing anything.
- Use the `python4poets` package (`import python4poets as p4p`) as documented in `README.md` in this folder, and follow its section "Notes for AI coding assistants".
- To check that the setup works at any time, run `python -m python4poets` with the `.venv` Python.

## Keep everything inside this folder

- Never use `sudo`, change system settings, or create or change files outside this folder. If something seems to need that, stop and explain the problem to the student instead of working around it.
- Keep all downloaded data inside this folder:
  - `python4poets` already does: its data goes in `p4p_data/`.
  - Stopwords: use `p4p.get_stopwords()`, which needs no download (not `nltk.download("stopwords")`).
  - Other NLTK data (e.g. the `punkt_tab` tokenizer for `nltk.word_tokenize`): use `p4p.download_nltk_data("punkt_tab")`, not `nltk.download(...)`. It saves the data in `p4p_data/nltk_data` and tells NLTK to look there.
  - Hugging Face datasets: put `os.environ["HF_HOME"] = "p4p_data/huggingface"` at the top of the script or notebook, before `import datasets`.
- Commands that download anything need network access: `pip install`, downloading files, and any Python code that loads a corpus, Gutenberg books, word vectors or a dataset (including `python -m python4poets`). Run them with network access enabled from the start, rather than trying without it first. If the network is still blocked, ask the student to allow it; don't look for another way round.

## Where code goes

- Write code in short script files in this folder, one per question, with descriptive names (e.g. `austen_vs_bronte.py`), not as inline commands (`python -c "..."`). Run them with the `.venv` Python, and tell the student the command so they can rerun it themselves.
- Start each script with a comment saying what question it answers. Keep it short; comment each step.
- Save plots as image files (`fig.save("austen_vs_bronte.png")`) and tables as CSV (`result.to_csv(...)`), and print only what's needed (e.g. `result.head(20)`).
- For slow steps (topic models, anything that takes more than a few seconds) or work with many plots, suggest a Jupyter notebook instead.
- One-off checks (e.g. whether a package is installed) can stay as commands.

## Teaching, not just doing

- Keep code short and readable, and explain what each step does, and why, in plain language.
- Prefer a few clear steps the student can follow over clever one-liners.
- When interpreting results (distinctive words, topics, word-vector neighbours), show examples in context with `p4p.kwic`, and let the student draw the conclusions.
- If something fails, show the student the error message and explain it before trying a fix.
