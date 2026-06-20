# Personal RAG Knowledge Base v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a tested local CLI RAG MVP with incremental Markdown/TXT/PDF indexing, Chroma and BM25 hybrid retrieval, ModelScope-backed embeddings and generation, citations, stats, and evaluation.

**Architecture:** Keep JSON/JSONL manifests authoritative and treat Chroma and BM25 as derived indexes. Use small adapters for providers and storage, combine vector and BM25 rankings with RRF, and keep every CLI path runnable in deterministic mock mode while offering an opt-in ModelScope API mode.

**Tech Stack:** Python 3.12, pytest, PyYAML, requests, pypdf, jieba, rank-bm25, chromadb.

---

## File Map

```text
pyproject.toml                         package metadata, dependencies, rag entrypoint
.gitignore                             generated data, caches, secrets
config/config.yaml                     checked-in defaults
personal_rag/cli.py                    CLI parsing and terminal formatting
personal_rag/config.py                 typed config loader and env overrides
personal_rag/core/schema.py            domain dataclasses
personal_rag/core/hashing.py           hashes and stable IDs
personal_rag/core/metadata.py          metadata normalization
personal_rag/ingest/scanner.py         file change detection
personal_rag/ingest/parsers.py         Markdown/TXT/PDF parsing
personal_rag/ingest/chunker.py         overlap-aware chunks
personal_rag/ingest/indexer.py         incremental indexing transaction
personal_rag/providers/embeddings.py   deterministic mock and ModelScope embeddings
personal_rag/providers/generators.py   grounded mock and ModelScope chat
personal_rag/storage/atomic.py         atomic file replacement helpers
personal_rag/storage/doc_store.py      file manifest
personal_rag/storage/chunk_store.py    chunk JSONL manifest
personal_rag/storage/embedding_cache.py model-bound vector cache
personal_rag/storage/vector_store.py   Chroma adapter
personal_rag/storage/bm25_store.py     persisted Jieba BM25 adapter
personal_rag/storage/stats_store.py    last successful indexing metadata
personal_rag/rag/fusion.py             reciprocal-rank fusion
personal_rag/rag/retriever.py          hybrid retrieval
personal_rag/rag/context.py            prompt and citation map
personal_rag/rag/citations.py          citation extraction and validation
personal_rag/rag/pipeline.py           ask flow
personal_rag/evaluation/evaluator.py   Hit@K, MRR, citation coverage
tests/                                 unit, integration, CLI, and opt-in live tests
examples/docs/                         interview demonstration corpus
eval/dataset.json                      small labeled evaluation set
README.md                              install, configuration, demo, architecture
```

### Task 1: Package Skeleton, Configuration, and Domain Types

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config/config.yaml`
- Create: `personal_rag/__init__.py`
- Create: `personal_rag/config.py`
- Create: `personal_rag/core/__init__.py`
- Create: `personal_rag/core/schema.py`
- Create: `personal_rag/core/hashing.py`
- Create: `personal_rag/core/metadata.py`
- Create: `tests/test_config.py`
- Create: `tests/core/test_hashing.py`
- Create: `tests/core/test_metadata.py`

- [ ] **Step 1: Write failing configuration and core helper tests**

```python
def test_environment_overrides_modelscope_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELSCOPE_API_KEY", "secret")
    config = load_config(write_config(tmp_path))
    assert config.provider.api_key == "secret"

def test_make_id_is_stable_and_path_separator_independent():
    assert make_doc_id(r"notes\rag.md") == make_doc_id("notes/rag.md")

def test_normalize_metadata_replaces_none_and_stringifies_complex_values():
    assert normalize_metadata({"page": None, "tags": ["rag"]}) == {
        "page": "",
        "tags": "['rag']",
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/test_config.py tests/core -q
```

Expected: collection fails because `personal_rag.config` and core modules do not exist.

- [ ] **Step 3: Implement typed settings, schemas, hashing, and normalization**

Implement immutable dataclasses:

```python
@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    doc_type: str
    content: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

Add equivalent `Chunk`, `RetrievedChunk`, `Citation`, `Answer`, `ScanResult`,
`IndexReport`, and `EvaluationReport` records. Use `path.replace("\\", "/")`
before stable SHA-256-based IDs. Load YAML defaults and override the provider
key from `MODELSCOPE_API_KEY`.

The checked-in defaults must include:

```yaml
chunk:
  size: 700
  overlap: 120
retrieval:
  top_k: 5
  vector_top_k: 10
  bm25_top_k: 10
provider:
  mode: mock
  base_url: https://api-inference.modelscope.cn/v1
  embedding_model: Qwen/Qwen3-Embedding-0.6B
  embedding_dimensions: 1024
  llm_model: Qwen/Qwen3-30B-A3B-Instruct-2507
  timeout_seconds: 90
  fallback_to_mock: true
storage:
  base_dir: data/storage
  cache_dir: data/cache
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/test_config.py tests/core -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml .gitignore config personal_rag tests
git commit -m "feat: scaffold RAG core and configuration"
```

### Task 2: Scanner, Parsers, and Chunker

**Files:**
- Create: `personal_rag/ingest/__init__.py`
- Create: `personal_rag/ingest/scanner.py`
- Create: `personal_rag/ingest/parsers.py`
- Create: `personal_rag/ingest/chunker.py`
- Create: `tests/ingest/test_scanner.py`
- Create: `tests/ingest/test_parsers.py`
- Create: `tests/ingest/test_chunker.py`

- [ ] **Step 1: Write failing scanner, parser, and chunker tests**

Cover these exact behaviors:

```python
def test_scan_files_classifies_added_modified_deleted_and_unchanged(tmp_path):
    ...
    result = scan_files(tmp_path, old_state)
    assert [p.name for p in result.added] == ["added.md"]
    assert [p.name for p in result.modified] == ["changed.txt"]
    assert result.deleted == ["deleted.pdf"]

def test_parse_markdown_preserves_heading_per_document_section(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Intro\nA\n## Retrieval\nB", encoding="utf-8")
    docs = parse_file(path, tmp_path)
    assert [doc.metadata["heading"] for doc in docs] == ["Intro", "Retrieval"]

def test_split_text_has_overlap_and_makes_forward_progress():
    chunks = split_text("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]

def test_build_chunks_produces_stable_ids():
    assert build_chunks(doc)[0].chunk_id == build_chunks(doc)[0].chunk_id
```

Create a one-page PDF fixture with `pypdf.PdfWriter` and verify its source path
and page metadata even when extracted text is empty; add a small generated
text-bearing PDF later in the CLI fixture task.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/ingest -q
```

Expected: import failures for missing ingest modules.

- [ ] **Step 3: Implement scanning, parsing, and chunking**

Use:

```python
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
```

`scan_files()` must persist and compare relative POSIX paths. Markdown parsing
must split at ATX headings while retaining heading text. PDF parsing returns one
document per non-empty page. `split_text()` validates
`0 <= overlap < chunk_size`, prefers paragraph/newline boundaries, and falls
back to exact character positions.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/ingest -q
```

Expected: all ingest tests pass.

- [ ] **Step 5: Commit**

```powershell
git add personal_rag/ingest tests/ingest
git commit -m "feat: add document ingestion pipeline"
```

### Task 3: Atomic Manifests and Caches

**Files:**
- Create: `personal_rag/storage/__init__.py`
- Create: `personal_rag/storage/atomic.py`
- Create: `personal_rag/storage/doc_store.py`
- Create: `personal_rag/storage/chunk_store.py`
- Create: `personal_rag/storage/embedding_cache.py`
- Create: `personal_rag/storage/stats_store.py`
- Create: `tests/storage/test_stores.py`

- [ ] **Step 1: Write failing persistence tests**

```python
def test_doc_store_round_trips_file_mapping(tmp_path):
    store = DocStore(tmp_path / "docs.json")
    store.save({"notes.md": {"content_hash": "abc", "doc_ids": ["doc_1"]}})
    assert store.load()["notes.md"]["doc_ids"] == ["doc_1"]

def test_chunk_store_replaces_chunks_for_a_document(tmp_path):
    store = ChunkStore(tmp_path / "chunks.jsonl")
    store.save_all([old_a, keep])
    store.replace_doc_chunks("doc_a", [new_a])
    assert [c.chunk_id for c in store.load_all()] == ["keep", "new_a"]

def test_embedding_cache_key_changes_with_model(tmp_path):
    cache = EmbeddingCache(tmp_path)
    assert cache.key("provider", "model-a", 3, "hash") != cache.key(
        "provider", "model-b", 3, "hash"
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/storage/test_stores.py -q
```

Expected: import failures for missing storage modules.

- [ ] **Step 3: Implement atomic stores**

Write temporary files in the destination directory, flush, then call
`os.replace()`. Serialize dataclasses with `asdict()`. The embedding cache stores
one JSON vector per model-bound cache key. `StatsStore` stores:

```json
{
  "last_indexed_at": "ISO-8601",
  "docs_root": "POSIX path",
  "embedding_mode": "mock",
  "embedding_model": "mock-hash-embedding"
}
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/storage/test_stores.py -q
```

Expected: all store tests pass.

- [ ] **Step 5: Commit**

```powershell
git add personal_rag/storage tests/storage/test_stores.py
git commit -m "feat: add atomic manifests and embedding cache"
```

### Task 4: Embedding and Generation Providers

**Files:**
- Create: `personal_rag/providers/__init__.py`
- Create: `personal_rag/providers/embeddings.py`
- Create: `personal_rag/providers/generators.py`
- Create: `tests/providers/test_embeddings.py`
- Create: `tests/providers/test_generators.py`

- [ ] **Step 1: Write failing provider tests**

Inject a callable HTTP transport so tests never need network:

```python
def test_mock_embeddings_are_deterministic_and_normalized():
    client = MockEmbeddingClient(dimensions=32)
    first = client.embed(["混合检索"])[0]
    second = client.embed(["混合检索"])[0]
    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)

def test_modelscope_embedding_accepts_nested_response():
    transport = FakeTransport({"data": {"data": [{"embedding": [0.1, 0.2]}]}})
    client = APIEmbeddingClient(..., transport=transport)
    assert client.embed(["text"]) == [[0.1, 0.2]]
    assert transport.last_json["encoding_format"] == "float"

def test_modelscope_chat_rejects_empty_choices():
    generator = APIGenerator(..., transport=FakeTransport({"choices": None}))
    with pytest.raises(ProviderError, match="empty"):
        generator.generate("prompt")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/providers -q
```

Expected: import failures for missing provider modules.

- [ ] **Step 3: Implement providers**

The mock embedding client hashes Jieba tokens into a fixed vector, applies
signed accumulation, then L2-normalizes. API embeddings post to `/embeddings`
and support both standard `data[0].embedding` and ModelScope's observed nested
`data.data[0].embedding`.

The mock generator extracts the first context block and returns a concise
answer ending in `[C1]`. API generation posts to `/chat/completions`, validates
that `choices[0].message.content` is non-empty, maps HTTP and network failures to
sanitized `ProviderError`, and never includes authorization values.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/providers -q
```

Expected: all provider tests pass.

- [ ] **Step 5: Commit**

```powershell
git add personal_rag/providers tests/providers
git commit -m "feat: add mock and ModelScope providers"
```

### Task 5: Chroma and BM25 Stores

**Files:**
- Create: `personal_rag/storage/vector_store.py`
- Create: `personal_rag/storage/bm25_store.py`
- Create: `tests/storage/test_vector_store.py`
- Create: `tests/storage/test_bm25_store.py`

- [ ] **Step 1: Install project dependencies**

Run:

```powershell
py -3.12 -m pip install -e ".[dev]"
```

Expected: installation succeeds and `chromadb`, `rank_bm25`, `jieba`, `pypdf`,
`yaml`, and `requests` import under Python 3.12.

- [ ] **Step 2: Write failing index adapter tests**

```python
def test_vector_store_upserts_queries_and_deletes(tmp_path):
    store = VectorStore(tmp_path / "chroma", "chunks")
    store.upsert([chunk], [[1.0, 0.0]])
    assert store.query([1.0, 0.0], top_k=1)[0].chunk_id == chunk.chunk_id
    store.delete([chunk.chunk_id])
    assert store.query([1.0, 0.0], top_k=1) == []

def test_bm25_store_ranks_exact_chinese_term(tmp_path):
    store = BM25Store(tmp_path / "bm25.pkl")
    store.build([rerank_chunk, unrelated_chunk])
    assert store.search("为什么需要精排", top_k=1)[0].chunk_id == "rerank"
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/storage/test_vector_store.py tests/storage/test_bm25_store.py -q
```

Expected: imports fail because adapters do not exist.

- [ ] **Step 4: Implement adapters**

Chroma uses `PersistentClient`, cosine space, supplied embeddings, normalized
primitive metadata, and document text. Convert cosine distance to
`1.0 - distance`.

BM25 persists chunk payloads and tokenized corpus, recreates `BM25Okapi` after
loading, filters zero-score results, and returns `RetrievedChunk` records.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/storage/test_vector_store.py tests/storage/test_bm25_store.py -q
```

Expected: all index adapter tests pass.

- [ ] **Step 6: Commit**

```powershell
git add personal_rag/storage pyproject.toml tests/storage
git commit -m "feat: add Chroma and BM25 indexes"
```

### Task 6: Incremental Indexing Orchestration

**Files:**
- Create: `personal_rag/ingest/indexer.py`
- Create: `tests/ingest/test_indexer.py`

- [ ] **Step 1: Write failing incremental integration tests**

Use temporary real manifests and BM25, with a recording embedding client and
temporary Chroma directory:

```python
def test_second_unchanged_index_skips_embedding(index_env):
    first = index_env.indexer.index(index_env.docs)
    calls_after_first = index_env.embedding.calls
    second = index_env.indexer.index(index_env.docs)
    assert first.added == 1
    assert second.unchanged == 1
    assert index_env.embedding.calls == calls_after_first

def test_modified_file_replaces_only_its_chunks(index_env):
    ...
    assert all(c.text != "old text" for c in chunks)
    assert any(c.doc_id == untouched_doc_id for c in chunks)

def test_deleted_file_removes_manifest_chunks_and_vectors(index_env):
    ...
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/ingest/test_indexer.py -q
```

Expected: import failure for missing `Indexer`.

- [ ] **Step 3: Implement the incremental transaction**

Build `next_chunks` and `next_doc_state` in memory. Reuse cached embeddings.
Perform targeted Chroma deletes/upserts. Build BM25 from `next_chunks`. Publish
chunk and document manifests only after provider and derived-index operations
succeed. Record stats last.

Return an `IndexReport` with scanned, added, modified, deleted, unchanged,
logical document, chunk, cache-hit, and skipped-PDF-page counts.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/ingest/test_indexer.py -q
```

Expected: all incremental indexing tests pass.

- [ ] **Step 5: Run all tests so far**

Run:

```powershell
py -3.12 -m pytest -q
```

Expected: all current tests pass.

- [ ] **Step 6: Commit**

```powershell
git add personal_rag/ingest/indexer.py tests/ingest/test_indexer.py
git commit -m "feat: implement incremental indexing"
```

### Task 7: Hybrid Retrieval, Context, Citations, and Ask Pipeline

**Files:**
- Create: `personal_rag/rag/__init__.py`
- Create: `personal_rag/rag/fusion.py`
- Create: `personal_rag/rag/retriever.py`
- Create: `personal_rag/rag/context.py`
- Create: `personal_rag/rag/citations.py`
- Create: `personal_rag/rag/pipeline.py`
- Create: `tests/rag/test_fusion.py`
- Create: `tests/rag/test_context_and_citations.py`
- Create: `tests/rag/test_pipeline.py`

- [ ] **Step 1: Write failing RAG tests**

```python
def test_rrf_rewards_chunks_present_in_both_rankings():
    fused = rrf_fusion([a, b], [b, c], top_k=3)
    assert fused[0].chunk_id == b.chunk_id
    assert fused[0].source == "fusion"

def test_checker_filters_unknown_citations():
    citations = check_citations("事实。[C1][C99]", {"C1": chunk})
    assert [citation.citation_id for citation in citations] == ["C1"]

def test_pipeline_returns_insufficient_without_results():
    answer = RAGPipeline(EmptyRetriever(), generator).ask("unknown")
    assert answer.answer == "当前知识库信息不足，无法确定。"
    assert answer.citations == []
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/rag -q
```

Expected: imports fail because RAG modules do not exist.

- [ ] **Step 3: Implement the online pipeline**

`HybridRetriever` embeds one query, asks Chroma and BM25 for configured
candidate counts, then applies RRF. `build_context()` creates `C1..Cn`, includes
source path, page, heading, and the exact chunk text. `check_citations()` keeps
first occurrence order and filters unknown identifiers.

When API generation fails and `fallback_to_mock` is enabled, call the mock
generator and set `Answer.generation_mode` to `"mock-fallback"`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/rag -q
```

Expected: all RAG tests pass.

- [ ] **Step 5: Commit**

```powershell
git add personal_rag/rag tests/rag
git commit -m "feat: add hybrid retrieval and cited answers"
```

### Task 8: Evaluation and CLI

**Files:**
- Create: `personal_rag/evaluation/__init__.py`
- Create: `personal_rag/evaluation/evaluator.py`
- Create: `personal_rag/cli.py`
- Create: `tests/evaluation/test_evaluator.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing evaluator and CLI tests**

```python
def test_reciprocal_rank_uses_first_expected_source():
    assert reciprocal_rank(results, ["expected.md"]) == pytest.approx(0.5)

def test_citation_coverage_counts_answers_with_legal_citations():
    assert citation_coverage([with_citation, without_citation]) == 0.5

def test_cli_exposes_all_commands(runner):
    assert runner(["--help"]).exit_code == 0
    for command in ["index", "search", "ask", "stats", "eval"]:
        assert command in runner(["--help"]).stdout
```

Add CLI integration assertions for output labels and non-zero missing-index
errors. Keep `main(argv: Sequence[str] | None = None) -> int` directly testable
without subprocesses.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/evaluation tests/test_cli.py -q
```

Expected: imports fail because evaluator and CLI do not exist.

- [ ] **Step 3: Implement evaluator and CLI composition root**

The CLI must resolve all storage paths relative to the config file/project
working directory, construct providers and stores, format Chinese-readable
summaries, and return:

```text
0 success
2 invalid arguments or missing input
3 missing/corrupt index
4 provider failure
1 unexpected application failure
```

`eval` runs retrieval and ask for each dataset item and reports all required
metrics.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/evaluation tests/test_cli.py -q
```

Expected: evaluator and CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add personal_rag/evaluation personal_rag/cli.py tests/evaluation tests/test_cli.py
git commit -m "feat: add evaluation and CLI commands"
```

### Task 9: Demonstration Corpus and Documentation

**Files:**
- Create: `examples/docs/rag_notes.md`
- Create: `examples/docs/rag_design.md`
- Create: `examples/docs/search_notes.txt`
- Create: `examples/docs/system_notes.pdf`
- Create: `eval/dataset.json`
- Create: `tests/test_acceptance_cli.py`
- Create: `tests/live/test_modelscope.py`
- Create: `README.md`

- [ ] **Step 1: Write failing five-command acceptance test**

In an isolated temporary working directory, copy the example corpus and config,
force mock mode, then run:

```python
assert cli(["index", str(docs)]) == 0
assert cli(["search", "RAG 为什么需要 Rerank？"]) == 0
assert cli(["ask", "这个系统怎么做增量索引？"]) == 0
assert cli(["stats"]) == 0
assert cli(["eval", str(dataset)]) == 0
```

Assert output includes `Chroma`, `BM25`, `source`, `[C1]`, `Chunk`, `Hit@1`,
`Hit@3`, `MRR`, and `引用覆盖率`.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
py -3.12 -m pytest tests/test_acceptance_cli.py -q
```

Expected: failure because examples, dataset, or documented fixture behavior is
not complete.

- [ ] **Step 3: Add corpus, dataset, README, and opt-in live test**

The example corpus must contain explicit passages about:

- why reranking follows first-stage retrieval
- BM25 and vector retrieval strengths
- RRF rank fusion
- content-hash incremental indexing
- embedding cache keys
- citation validation

Generate a text-bearing PDF with a small helper during development and commit
the resulting PDF fixture. The live test is decorated:

```python
@pytest.mark.live
@pytest.mark.skipif(not os.getenv("MODELSCOPE_API_KEY"), reason="token not set")
def test_modelscope_embedding_and_chat():
    ...
```

README includes Python 3.12 installation, user environment variable setup,
mock/API config switching, the five demo commands, architecture explanation,
limitations, and troubleshooting for 401/429/timeouts.

- [ ] **Step 4: Run acceptance test and verify GREEN**

Run:

```powershell
py -3.12 -m pytest tests/test_acceptance_cli.py -q
```

Expected: five-command acceptance test passes.

- [ ] **Step 5: Commit**

```powershell
git add examples eval tests/test_acceptance_cli.py tests/live README.md
git commit -m "docs: add interview demo corpus and guide"
```

### Task 10: Final Verification and Live ModelScope Check

**Files:**
- Modify only files required by failures discovered during verification.

- [ ] **Step 1: Run formatting/static checks**

Run:

```powershell
py -3.12 -m ruff check .
```

Expected: zero errors.

- [ ] **Step 2: Run the complete offline test suite**

Run:

```powershell
py -3.12 -m pytest -m "not live" -q
```

Expected: zero failures.

- [ ] **Step 3: Install the console command and run all acceptance commands**

Run:

```powershell
py -3.12 -m pip install -e .
rag index ./examples/docs
rag search "RAG 为什么需要 Rerank？"
rag ask "这个系统怎么做增量索引？"
rag stats
rag eval ./eval/dataset.json
```

Expected: all commands exit zero and print the required evidence.

- [ ] **Step 4: Verify unchanged incremental behavior**

Run:

```powershell
rag index ./examples/docs
```

Expected: all supported files are reported unchanged, with zero new embedding
requests and no chunk-count change.

- [ ] **Step 5: Run the opt-in ModelScope smoke test**

Run:

```powershell
py -3.12 -m pytest tests/live/test_modelscope.py -m live -q
```

Expected: one embedding call returns 1024 dimensions and one chat call returns
non-empty content.

- [ ] **Step 6: Review requirements and repository state**

Run:

```powershell
git status --short
git log --oneline --decorate -10
```

Expected: only intentional final changes are present; every v0.1 acceptance
criterion maps to passing evidence.

- [ ] **Step 7: Commit final fixes**

```powershell
git add -A
git commit -m "test: verify personal RAG v0.1 acceptance"
```

