# Personal RAG Knowledge Base v0.1.2 Design

## Objective

Freeze the v0.1 interview-ready line with automated public verification and a
short interviewer-oriented project explanation.

No RAG behavior, storage format, retrieval algorithm, provider contract, or CLI
semantics change in this release.

## GitHub Actions CI

Add `.github/workflows/ci.yml`.

The workflow runs on:

```text
push to main
pull requests targeting main
manual workflow dispatch
```

Use a Python matrix:

```text
3.11
3.12
```

Each matrix job:

1. checks out the repository;
2. installs the requested Python version with pip caching;
3. upgrades pip;
4. installs `-e ".[dev]"`;
5. runs `python -m ruff check .`;
6. runs `python -m pytest -m "not live" -q`;
7. builds a wheel with `python -m pip wheel . --no-deps --wheel-dir dist`.

ModelScope live tests are deliberately excluded because CI must not require a
user API key and should remain deterministic.

The workflow uses minimal permissions:

```yaml
permissions:
  contents: read
```

Concurrency cancels stale runs for the same branch or pull request.

## README Interview Section

Add `## 面试讲解重点` after architecture/data flow and before operational
details.

The section explains the project in four concise points:

- offline indexing: file hash, parsing, chunks, Chroma and BM25;
- online answering: two-stage recall, RRF, cited context and validation;
- engineering reliability: incremental indexing, cache-preserving rebuild and
  ModelScope/Mock modes;
- evaluation: Hit@K, MRR, citation coverage and automated acceptance tests.

Add a CI badge near the title linking to the GitHub Actions workflow. Keep the
existing architecture image and demo output unchanged.

## Version Metadata

Update:

```text
pyproject.toml version: 0.1.2
personal_rag.__version__: 0.1.2
CHANGELOG.md: 0.1.2 entry
```

The changelog records:

- GitHub Actions on Python 3.11 and 3.12;
- README interview explanation and CI badge;
- no runtime behavior changes.

## Testing

Add repository metadata tests that verify:

- package and module versions both equal `0.1.2`;
- `.github/workflows/ci.yml` exists;
- workflow text includes Python 3.11/3.12, Ruff, non-live pytest and wheel
  build commands;
- README includes the CI badge and `面试讲解重点`;
- the interview section mentions offline indexing, RRF, rebuild, Hit@K and
  MRR.

Local final verification:

```text
Ruff
all non-live pytest tests
ModelScope live smoke test
wheel build
```

After push, verify the GitHub Actions run finishes successfully before creating
the final `v0.1.2` release.

## Release

```text
tag: v0.1.2
release title: Personal RAG Knowledge Base v0.1.2
```

After v0.1.2, freeze v0.1 except for genuine defects. New RAG capabilities
belong in v0.2.

