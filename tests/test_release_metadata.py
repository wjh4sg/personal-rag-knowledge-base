import tomllib
from pathlib import Path

import personal_rag

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pyproject():
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_metadata_is_consistent():
    pyproject = load_pyproject()

    assert pyproject["project"]["version"] == "0.1.1"
    assert personal_rag.__version__ == "0.1.1"


def test_project_uses_mit_license():
    pyproject = load_pyproject()
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert pyproject["project"]["license"] == "MIT"
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 wjh4sg" in license_text


def test_readme_embeds_architecture_and_demo_outputs():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (PROJECT_ROOT / "docs" / "architecture.svg").read_text(
        encoding="utf-8"
    )

    assert "docs/architecture.svg" in readme
    assert "## Demo 输出示例" in readme
    assert "rag rebuild" in readme
    assert "source: fusion" in readme
    assert "<title>" in architecture
    assert "<desc>" in architecture
