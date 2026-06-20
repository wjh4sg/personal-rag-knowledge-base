# Personal RAG Knowledge Base v0.1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish v0.1.1 with complete index validation, a cache-preserving `rag rebuild`, stronger repository presentation, and MIT licensing.

**Architecture:** Keep the existing `Indexer` as the sole indexing pipeline. Add narrow reset operations to configured storage adapters, orchestrate rebuild in the CLI after validating the source directory, and keep embedding cache storage outside the reset set.

**Tech Stack:** Python 3.12, argparse, Chroma, rank-bm25, pytest, Ruff, SVG, setuptools.

---

### Task 1: Complete Index Validation

**Files:**
- Modify: `personal_rag/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing incomplete-vector-index test**

```python
def test_cli_rejects_incomplete_index_when_chroma_is_empty(tmp_path, capsys):
    config = write_config(tmp_path)
    base = tmp_path / "storage"
    ChunkStore(base / "chunks.jsonl").save_all([make_chunk()])
    bm25 = BM25Store(base / "bm25.pkl")
    bm25.build([make_chunk()])
    bm25.save()

    exit_code = main(["--config", str(config), "search", "query"])

    assert exit_code == 3
    assert "索引不存在或不完整" in capsys.readouterr().err
    assert "rag rebuild" in capsys.readouterr().err
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/test_cli.py::test_cli_rejects_incomplete_index_when_chroma_is_empty -q
```

Expected: FAIL because `_require_index` currently accepts Chunk and BM25 files without a non-empty Chroma collection.

- [ ] **Step 3: Implement complete validation**

Change `_require_index` to require:

```python
if (
    not services.chunk_store.exists()
    or not services.bm25_store.exists()
    or not services.vector_store.exists()
):
    raise MissingIndexError(
        "知识库索引不存在或不完整，请先运行 rag index <文档目录>，"
        "或运行 rag rebuild <文档目录> 全量重建。"
    )
```

Load BM25 only after this check.

- [ ] **Step 4: Run targeted and CLI tests**

Run:

```powershell
py -3.12 -m pytest tests/test_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add personal_rag/cli.py tests/test_cli.py
git commit -m "fix: detect incomplete retrieval indexes"
```

### Task 2: Cache-Preserving Rebuild

**Files:**
- Modify: `personal_rag/storage/doc_store.py`
- Modify: `personal_rag/storage/chunk_store.py`
- Modify: `personal_rag/storage/bm25_store.py`
- Modify: `personal_rag/storage/stats_store.py`
- Modify: `personal_rag/storage/vector_store.py`
- Modify: `personal_rag/cli.py`
- Modify: `tests/storage/test_stores.py`
- Modify: `tests/storage/test_bm25_store.py`
- Modify: `tests/storage/test_vector_store.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_acceptance_cli.py`

- [ ] **Step 1: Write failing reset tests for stores**

Add tests proving:

```python
def test_doc_store_reset_removes_manifest(tmp_path):
    store = DocStore(tmp_path / "docs.json")
    store.save({"notes.md": {"doc_ids": ["doc_1"]}})
    store.reset()
    assert not store.exists()

def test_bm25_reset_removes_pickle_and_in_memory_index(tmp_path):
    ...
    store.reset()
    assert not store.exists()
    assert store.search("query", 5) == []

def test_vector_reset_removes_all_vectors_and_recreates_collection(tmp_path):
    ...
    store.reset()
    assert not store.exists()
    store.upsert([chunk], [[1.0, 0.0]])
    assert store.exists()
```

- [ ] **Step 2: Run reset tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/storage -q
```

Expected: FAIL because reset methods do not exist.

- [ ] **Step 3: Implement narrow reset methods**

Manifest and stats stores use:

```python
def reset(self) -> None:
    self.path.unlink(missing_ok=True)
```

BM25 additionally clears:

```python
self.chunks = []
self.corpus_tokens = []
self.index = None
```

Vector reset must:

1. delete the collection if it exists;
2. release client references;
3. remove only `self.directory`;
4. recreate the persistent client and named collection.

Store `collection_name` on the adapter and centralize initialization in
`_connect()`.

- [ ] **Step 4: Write failing rebuild CLI tests**

Cover:

```python
def test_cli_exposes_rebuild(capsys):
    ...
    assert "rebuild" in output

def test_rebuild_validates_docs_before_reset(tmp_path, capsys):
    ...
    assert main([... "rebuild", str(missing)]) == 2
    assert old_manifest.exists()

def test_rebuild_clears_stale_state_preserves_cache_and_unrelated_files(...):
    ...
    assert "Embedding 缓存命中：" in output
    assert unrelated.read_text() == "keep"
    assert cached_embedding_file.exists()
```

The integration test first indexes the demo corpus, writes a stale manifest
record and an unrelated storage file, runs rebuild, then verifies the rebuilt
files only contain current docs and the cache directory is unchanged.

- [ ] **Step 5: Run rebuild tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/test_cli.py tests/test_acceptance_cli.py -q
```

Expected: FAIL because `rebuild` is not registered.

- [ ] **Step 6: Implement rebuild orchestration**

Add:

```python
def reset_index(services: Services) -> None:
    services.doc_store.reset()
    services.chunk_store.reset()
    services.bm25_store.reset()
    services.stats_store.reset()
    services.vector_store.reset()
```

Add `run_rebuild()` that validates `docs_path.is_dir()` before reset, prints a
reset message, then calls `run_index()`.

Register:

```text
rag rebuild <docs_path>
```

in argparse and command dispatch.

- [ ] **Step 7: Run targeted and full offline tests**

Run:

```powershell
py -3.12 -m pytest tests/storage tests/test_cli.py tests/test_acceptance_cli.py -q
py -3.12 -m pytest -m "not live" -q
```

Expected: all tests pass, including the six-command acceptance flow.

- [ ] **Step 8: Commit**

```powershell
git add personal_rag/storage personal_rag/cli.py tests
git commit -m "feat: add cache-preserving index rebuild"
```

### Task 3: Demo Documentation, Architecture SVG, and MIT License

**Files:**
- Create: `docs/architecture.svg`
- Create: `LICENSE`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `personal_rag/__init__.py`
- Modify: `CHANGELOG.md`
- Create: `tests/test_release_metadata.py`

- [ ] **Step 1: Write failing release metadata tests**

```python
def test_version_metadata_is_consistent():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "0.1.1"
    assert personal_rag.__version__ == "0.1.1"

def test_project_uses_mit_license():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"] == "MIT"
    assert "MIT License" in Path("LICENSE").read_text(encoding="utf-8")

def test_readme_embeds_architecture_and_demo_outputs():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/architecture.svg" in readme
    assert "## Demo 输出示例" in readme
    assert "rag rebuild" in readme
    assert "source: fusion" in readme
```

- [ ] **Step 2: Run metadata tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/test_release_metadata.py -q
```

Expected: FAIL because version remains 0.1.0 and license/SVG/demo sections are absent.

- [ ] **Step 3: Add MIT license and version metadata**

Add the standard MIT text. Change:

```toml
[build-system]
requires = ["setuptools>=77"]

[project]
version = "0.1.1"
license = "MIT"
```

Change `personal_rag.__version__` to `0.1.1`.

- [ ] **Step 4: Add architecture.svg**

Create a self-contained 1200x720 SVG with:

- source document lane;
- offline indexing lane;
- Chroma/BM25 split and RRF merge;
- generation and citation lane;
- evaluation/stats observer lane;
- accessible `<title>` and `<desc>`;
- no external fonts, scripts, or images.

- [ ] **Step 5: Add exact demo outputs and rebuild docs**

Use outputs produced by the mock demo corpus. Include concise blocks for all six
commands and update troubleshooting to recommend `rag rebuild` instead of
manual deletion.

Add the diagram:

```markdown
![Personal RAG 架构图](docs/architecture.svg)
```

- [ ] **Step 6: Update changelog**

Add:

```markdown
## [0.1.1] - 2026-06-21

### Added
- cache-preserving `rag rebuild`
- architecture SVG and real demo output
- MIT license

### Fixed
- retrieval commands now reject partial indexes
```

Add the `v0.1.1` release link and retain `0.1.0`.

- [ ] **Step 7: Verify package metadata and docs**

Run:

```powershell
py -3.12 -m pytest tests/test_release_metadata.py -q
py -3.12 -m pip wheel . --no-deps --wheel-dir dist
```

Expected: tests pass and a `personal_rag-0.1.1` wheel builds.

- [ ] **Step 8: Commit**

```powershell
git add LICENSE README.md CHANGELOG.md docs/architecture.svg pyproject.toml personal_rag/__init__.py tests/test_release_metadata.py
git commit -m "docs: prepare v0.1.1 release"
```

### Task 4: Final Verification and Release

**Files:**
- Modify only files required by verification failures.

- [ ] **Step 1: Run static and offline verification**

```powershell
py -3.12 -m ruff check .
py -3.12 -m pytest -m "not live" -q
py -3.12 -m compileall -q personal_rag
py -3.12 -m pip check
```

Expected: zero failures and no broken requirements.

- [ ] **Step 2: Run ModelScope smoke test**

```powershell
$env:MODELSCOPE_API_KEY = [Environment]::GetEnvironmentVariable('MODELSCOPE_API_KEY', 'User')
py -3.12 -m pytest tests/live/test_modelscope.py -m live -q
```

Expected: one live test passes.

- [ ] **Step 3: Run six CLI commands**

```powershell
rag index ./examples/docs
rag search "RAG 为什么需要 Rerank？"
rag ask "这个系统怎么做增量索引？"
rag stats
rag eval ./eval/dataset.json
rag rebuild ./examples/docs
```

Expected: all exit zero; rebuild reports cache hits.

- [ ] **Step 4: Verify clean repository and commit any final fixes**

```powershell
git diff --check
git status --short
git log -5 --oneline
```

- [ ] **Step 5: Merge, push, tag, and release**

After merged-main verification:

```powershell
git push origin main
git tag -a v0.1.1 -m "Personal RAG Knowledge Base v0.1.1"
git push origin v0.1.1
gh release create v0.1.1 --title "Personal RAG Knowledge Base v0.1.1" --notes-file CHANGELOG.md --verify-tag
```

Expected: the public repository default branch and release page expose v0.1.1.

