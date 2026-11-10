import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_methodology_notebook_matches_document():
    """notebooks/methodology.ipynb is generated from METHODOLOGY.md and must not drift."""
    spec = importlib.util.spec_from_file_location(
        "build_methodology_notebook", ROOT / "scripts" / "build_methodology_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / "notebooks" / "methodology.ipynb").read_text() == module.render(), (
        "Run: python scripts/build_methodology_notebook.py")
