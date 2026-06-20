# Personal RAG Knowledge Base v0.1.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic GitHub Actions verification and an interviewer-oriented README section, then publish the frozen v0.1.2 interview release.

**Architecture:** CI mirrors the existing local quality gates and explicitly excludes credentialed live tests. Repository metadata tests enforce workflow contents, README messaging, and version consistency.

**Tech Stack:** GitHub Actions, Python 3.11/3.12, pytest, Ruff, setuptools.

---

### Task 1: CI and Interview Documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `tests/test_release_metadata.py`

- [ ] **Step 1: Write failing metadata tests**

Add:

```python
def test_ci_workflow_covers_supported_python_and_quality_gates():
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '"3.11"' in workflow
    assert '"3.12"' in workflow
    assert 'python -m ruff check .' in workflow
    assert 'python -m pytest -m "not live" -q' in workflow
    assert "python -m pip wheel . --no-deps --wheel-dir dist" in workflow

def test_readme_has_ci_badge_and_interview_talking_points():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "## 面试讲解重点" in readme
    for phrase in ["离线索引", "RRF", "rebuild", "Hit@K", "MRR"]:
        assert phrase in readme
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/test_release_metadata.py -q
```

Expected: FAIL because the workflow and interview section do not exist.

- [ ] **Step 3: Add the workflow**

Create `.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m ruff check .
      - run: python -m pytest -m "not live" -q
      - run: python -m pip wheel . --no-deps --wheel-dir dist
```

- [ ] **Step 4: Add README content**

Add the badge directly under the title:

```markdown
[![CI](https://github.com/wjh4sg/personal-rag-knowledge-base/actions/workflows/ci.yml/badge.svg)](https://github.com/wjh4sg/personal-rag-knowledge-base/actions/workflows/ci.yml)
```

Add `## 面试讲解重点` with four bullets covering offline indexing, online RRF
and citations, engineering reliability including rebuild/cache/provider modes,
and evaluation including Hit@K/MRR/acceptance tests.

- [ ] **Step 5: Run metadata and full offline tests**

```powershell
py -3.12 -m pytest tests/test_release_metadata.py -q
py -3.12 -m pytest -m "not live" -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add .github/workflows/ci.yml README.md tests/test_release_metadata.py
git commit -m "ci: verify supported Python versions"
```

### Task 2: Version and Release

**Files:**
- Modify: `pyproject.toml`
- Modify: `personal_rag/__init__.py`
- Modify: `CHANGELOG.md`
- Modify: `tests/test_release_metadata.py`

- [ ] **Step 1: Change version expectation and verify RED**

Change metadata test expectations from `0.1.1` to `0.1.2`.

Run:

```powershell
py -3.12 -m pytest tests/test_release_metadata.py::test_version_metadata_is_consistent -q
```

Expected: FAIL because package metadata remains 0.1.1.

- [ ] **Step 2: Update version and changelog**

Set both versions to `0.1.2`. Add changelog:

```markdown
## [0.1.2] - 2026-06-21

### Added
- GitHub Actions verification on Python 3.11 and 3.12.
- README CI badge and interview talking points.

### Changed
- v0.1 is now frozen except for genuine defects.
```

- [ ] **Step 3: Run complete local verification**

```powershell
py -3.12 -m ruff check .
py -3.12 -m pytest -m "not live" -q
$env:MODELSCOPE_API_KEY = [Environment]::GetEnvironmentVariable('MODELSCOPE_API_KEY', 'User')
py -3.12 -m pytest tests/live/test_modelscope.py -m live -q
py -3.12 -m pip wheel . --no-deps --wheel-dir dist
```

Expected: all checks pass and `personal_rag-0.1.2` wheel builds.

- [ ] **Step 4: Commit**

```powershell
git add pyproject.toml personal_rag/__init__.py CHANGELOG.md tests/test_release_metadata.py
git commit -m "docs: prepare v0.1.2 release"
```

- [ ] **Step 5: Merge and push**

Fast-forward the feature branch to `main`, repeat Ruff and non-live pytest on
merged main, then push `main`.

- [ ] **Step 6: Verify GitHub Actions**

Use:

```powershell
gh run list --workflow ci.yml --branch main --limit 1
gh run watch <run-id> --exit-status
```

Expected: Python 3.11 and 3.12 matrix jobs succeed.

- [ ] **Step 7: Tag and release**

```powershell
git tag -a v0.1.2 -m "Personal RAG Knowledge Base v0.1.2"
git push origin v0.1.2
gh release create v0.1.2 --title "Personal RAG Knowledge Base v0.1.2" --notes "Adds GitHub Actions verification on Python 3.11/3.12 and interviewer-oriented project documentation. The v0.1 interview line is now frozen except for genuine defects." --verify-tag
```

