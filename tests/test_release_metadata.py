import tomllib
from pathlib import Path

import personal_rag

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pyproject():
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_metadata_is_consistent():
    pyproject = load_pyproject()

    assert pyproject["project"]["version"] == "0.1.2"
    assert personal_rag.__version__ == "0.1.2"


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


def test_ci_workflow_covers_supported_python_and_quality_gates():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    assert '"3.11"' in workflow
    assert '"3.12"' in workflow
    assert "python -m ruff check ." in workflow
    assert 'python -m pytest -m "not live" -q' in workflow
    assert "python -m pip wheel . --no-deps --wheel-dir dist" in workflow


def test_readme_has_ci_badge_and_interview_talking_points():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "## 面试讲解重点" in readme
    for phrase in ["离线索引", "RRF", "rebuild", "Hit@K", "MRR"]:
        assert phrase in readme
