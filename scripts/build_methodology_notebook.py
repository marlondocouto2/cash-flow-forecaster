"""Generate notebooks/methodology.ipynb from METHODOLOGY.md (one markdown cell per section).

    python scripts/build_methodology_notebook.py

METHODOLOGY.md is the source of truth; a test checks the notebook is up to date.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "METHODOLOGY.md"
TARGET = ROOT / "notebooks" / "methodology.ipynb"


def _relink(text: str) -> str:
    """Rewrite repo-root relative links so they work from the notebooks/ folder."""
    def fix(match: re.Match) -> str:
        target = match.group(1)
        if re.match(r"^[a-z]+://|^#", target):
            return match.group(0)
        if target.startswith("notebooks/"):
            return f"]({target.removeprefix('notebooks/')})"
        return f"](../{target})"
    return re.sub(r"\]\(([^)]+)\)", fix, text)


def _jupyter_math(text: str) -> str:
    """GitHub's math-safe syntax -> the $...$ / $$...$$ syntax Jupyter understands.

    METHODOLOGY.md uses $`...`$ and ```math blocks because GitHub otherwise reads the
    underscores in subscripts as italics before rendering the math.
    """
    # Currency amounts are escaped as \$ in the .md. Notebook viewers (GitHub's, PyCharm's) don't all
    # honour that escape and would pair a dollar sign with the next formula, so write "USD" instead.
    text = re.sub(r"\\\$ ?(?=\w)", "USD ", text)  # "\$10M" -> "USD 10M", "\$ millions" -> "USD millions"
    text = text.replace("\\$", "USD")                  # "to \$." -> "to USD."
    text = re.sub(r"(?m)^([ \t]*)```math\n\1(.+?)\n\1```[ \t]*$", r"\1$$\2$$", text)
    return re.sub(r"\$`(.+?)`\$", r"$\1$", text)


def build_notebook(markdown: str) -> dict:
    """Split the document at level-2 headings into markdown cells."""
    sections = re.split(r"\n(?=## )", _jupyter_math(_relink(markdown)).strip() + "\n")
    cells = [
        {"cell_type": "markdown", "id": f"section-{i:02d}", "metadata": {},
         "source": section.rstrip("\n").splitlines(True)}
        for i, section in enumerate(sections)
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build_notebook(SOURCE.read_text()), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    TARGET.write_text(render())
    print(f"Wrote {TARGET.relative_to(ROOT)}")
