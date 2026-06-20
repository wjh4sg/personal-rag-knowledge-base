# Personal RAG Knowledge Base v0.1 Design

## 1. Objective

Build an interview-ready local CLI RAG system that indexes Markdown, TXT, and
text-based PDF files, performs hybrid retrieval, and generates answers with
validated citations.

The delivered CLI must support:

```text
rag index ./examples/docs
rag search "RAG 为什么需要 Rerank？"
rag ask "这个系统怎么做增量索引？"
rag stats
rag eval ./eval/dataset.json
```

The project is a local MVP. It does not include a Web UI, authentication,
multi-user support, OCR, agents, distributed storage, or production SLA.

## 2. Technical Decisions

### 2.1 Runtime and packaging

- Python 3.11 or newer.
- `pyproject.toml` defines the `rag` console script.
- Business code lives in a `personal_rag` package.
- Configuration is loaded from `config/config.yaml`, then overridden by
  environment variables where appropriate.
- Tests use `pytest`.

### 2.2 External services

ModelScope is the first real API provider:

```text
Base URL: https://api-inference.modelscope.cn/v1
Authentication: MODELSCOPE_API_KEY
Embedding model: Qwen/Qwen3-Embedding-0.6B
Embedding dimensions: 1024
LLM model: Qwen/Qwen3-30B-A3B-Instruct-2507
```

Embedding requests explicitly send `encoding_format: "float"`.

The API key is only read from the environment. It is never placed in YAML,
source code, logs, test fixtures, or committed files.

The provider interface remains OpenAI-compatible so another API can be selected
by changing the base URL and model names.

### 2.3 Offline behavior

The application supports explicit `mock` and `api` modes.

- `mock` is the default for automated tests and works without network access.
- `api` uses ModelScope and reports a clear actionable error when credentials,
  quota, network, moderation, or provider availability prevent a request.
- API failures do not silently change retrieval semantics.
- For the `ask` command only, configuration may enable an explicit
  `fallback_to_mock` option so an interview demo can continue when generation
  is unavailable. The CLI labels fallback output as mock-generated.

Mock embeddings are deterministic normalized hashed token vectors. They are
good enough to exercise Chroma and the complete vector path, but are never
presented as a quality substitute for the real embedding model.

## 3. Architecture

The system has three flows.

### 3.1 Offline indexing

```text
docs directory
  -> recursive scanner and SHA-256 file hashes
  -> parser registry
  -> Document records
  -> stable Chunk records
  -> embedding cache
  -> Chroma vector collection
  -> persisted BM25 index
  -> atomic document/chunk state update
```

Supported parsers:

- Markdown: UTF-8 text with current heading metadata retained per section.
- TXT: UTF-8 text with replacement for invalid bytes.
- PDF: text extraction with `pypdf`, one `Document` per non-empty page.

Images are not accepted in v0.1. This keeps implementation consistent with the
stated non-goal that OCR is deferred to v0.2.

### 3.2 Online retrieval and answering

```text
query
  -> lightweight normalization
  -> vector top-N retrieval
  -> Jieba-tokenized BM25 top-N retrieval
  -> Reciprocal Rank Fusion
  -> top-K context with C1..Cn identifiers
  -> LLM or mock generation
  -> citation extraction and validation
  -> Answer
```

Vector and BM25 raw scores are never added together. RRF combines ranks using
`1 / (60 + rank)`.

The generator is instructed to answer only from supplied chunks and to return
an insufficiency response when evidence is inadequate. Citation validation only
accepts identifiers assigned to the current retrieval result.

### 3.3 Evaluation and observability

`rag eval` runs the same production retriever and answer pipeline against a
JSON dataset. It reports:

- Hit@1
- Hit@3
- MRR
- citation coverage
- number of answers without citations

`rag stats` reports indexed file count, logical document count, chunk count,
index availability, embedding mode/model, and most recent successful indexing
time.

## 4. Module Boundaries

```text
personal_rag/
  cli.py                    argument parsing and formatted terminal output
  config.py                 typed configuration and environment overrides
  core/
    schema.py               Document, Chunk, RetrievedChunk, Citation, Answer
    hashing.py              stable IDs and SHA-256 helpers
    metadata.py             Chroma-safe metadata normalization
  ingest/
    scanner.py              added/modified/deleted/unchanged detection
    parsers.py              Markdown, TXT, and PDF parsing
    chunker.py              stable overlap-aware chunk construction
    indexer.py              indexing orchestration and consistency boundary
  providers/
    embeddings.py           mock and OpenAI-compatible embedding clients
    generators.py           mock and OpenAI-compatible chat clients
  storage/
    atomic.py               atomic JSON/JSONL/pickle replacement
    doc_store.py            file state and page-document mappings
    chunk_store.py          authoritative chunk JSONL store
    embedding_cache.py      model-bound embedding cache
    vector_store.py         Chroma adapter
    bm25_store.py           Jieba BM25 adapter
    stats_store.py          last successful index metadata
  rag/
    fusion.py               RRF
    retriever.py            hybrid retrieval
    context.py              prompt and citation map construction
    citations.py            structural citation validation
    pipeline.py             ask orchestration
  evaluation/
    evaluator.py            retrieval and citation metrics
```

Adapters expose small interfaces so provider and storage behavior can be tested
without coupling the pipeline to requests, Chroma internals, or filesystem
formats.

## 5. Data and Identity

### 5.1 File state

`docs.json` is keyed by POSIX-style path relative to the indexed root:

```json
{
  "rag_design.pdf": {
    "file_id": "file_...",
    "content_hash": "...",
    "doc_ids": ["doc_..._p1", "doc_..._p2"],
    "updated_at": "2026-06-21T12:00:00+08:00"
  }
}
```

No absolute source path is persisted.

### 5.2 Stable identifiers

- `file_id` is derived from normalized relative path.
- Text `doc_id` is derived from relative path.
- PDF page `doc_id` is derived from relative path and page number.
- `chunk_id` is derived from `doc_id` and zero-based chunk position.
- `chunk_hash` is SHA-256 of the exact chunk text.

This makes unchanged content stable while allowing modified documents to replace
their old chunks predictably.

### 5.3 Authoritative state

`chunks.jsonl` is the authoritative record for retrieval content and metadata.
Chroma and BM25 are derived indexes and can be rebuilt from it.

JSON, JSONL, pickle, and index metadata writes use temporary files followed by
atomic replacement. Indexing computes the next state before publishing document
and chunk manifests. If a provider request fails, the old manifests remain
usable.

## 6. Chunking

Defaults:

```text
chunk_size = 700 characters
overlap = 120 characters
```

The splitter prefers paragraph and line boundaries near the target size. It
falls back to character boundaries for long unbroken text. It guarantees
forward progress and rejects invalid settings such as negative overlap or
`overlap >= chunk_size`.

Markdown parsing tracks headings, and each chunk carries:

```text
source_path
title
page
heading
```

## 7. Incremental Indexing

The scanner compares current file hashes with `docs.json`.

- Added file: parse, chunk, embed, and add.
- Modified file: build replacement documents/chunks, remove all old file
  chunks, then add replacements.
- Deleted file: remove its document mappings, chunks, and vectors.
- Unchanged file: perform no parsing or embedding.

Embeddings are cached by:

```text
sha256(embedding provider + model + dimensions + chunk_hash)
```

BM25 is rebuilt from all current chunks after a successful change set. Chroma
receives targeted deletes and upserts. Running `index` with no changes does not
re-embed chunks and does not rebuild derived indexes unnecessarily.

## 8. Error Handling

CLI errors are concise, actionable, and return non-zero exit codes.

- Missing docs directory: identify the path and suggest creating or correcting
  it.
- Unsupported files: ignore them and include their count in the summary.
- Empty/scanned PDF pages: skip them and report skipped page count.
- No index: tell the user to run `rag index`.
- Missing API key in API mode: name `MODELSCOPE_API_KEY`.
- Provider 401/403: report authentication/authorization failure.
- Provider 429: report free quota or rate-limit exhaustion.
- Timeout/network failure: report the endpoint and configured timeout.
- Empty provider response: treat it as provider failure, not a valid answer.
- Corrupt derived index: report it and instruct the user to re-run `rag index`;
  source manifests remain readable.

Secret values and authorization headers are never included in exceptions shown
to users.

## 9. Testing Strategy

Implementation follows red-green-refactor.

Unit tests cover:

- hashing and path normalization
- scanner state transitions
- Markdown/TXT/PDF parsing
- chunk overlap, boundaries, and stable IDs
- metadata normalization
- embedding cache keys and cache hits
- ModelScope embedding/chat response parsing
- BM25 tokenization and ranking
- RRF duplicate fusion
- context construction
- citation filtering
- Hit@K, MRR, and citation coverage

Integration tests cover:

- first index
- no-change second index
- one-file modification
- deletion
- vector plus BM25 search
- answer generation with legal and illegal citations
- stats
- evaluation

CLI smoke tests run all five acceptance commands in mock mode. A separate,
opt-in live test uses `MODELSCOPE_API_KEY` and makes one minimal embedding and
one minimal chat request; it is excluded from the default test suite.

## 10. Acceptance Criteria

The implementation is accepted when:

1. A clean environment can install the package and expose `rag`.
2. `rag index ./examples/docs` indexes Markdown, TXT, and PDF content.
3. A second unchanged index run skips all unchanged files and makes no
   embedding calls.
4. Modifying or deleting one file only replaces/removes data owned by that file.
5. `rag search` returns fused results with score, preview, and source metadata.
6. `rag ask` returns a grounded answer and only valid citations, or the explicit
   insufficiency response.
7. `rag stats` exposes the required state.
8. `rag eval` prints Hit@1, Hit@3, MRR, and citation coverage.
9. The full test suite passes without requiring network access.
10. The opt-in ModelScope smoke test succeeds with the configured user-level
    `MODELSCOPE_API_KEY`.

## 11. Deferred Work

The following remain v0.2:

- OCR and image ingestion
- scanned and layout-heavy PDF handling
- semantic or heading-aware recursive chunk optimization beyond the simple
  boundary preference
- cross-encoder reranking
- RAGAS or LLM-as-judge faithfulness
- Web UI
- SQLite migration
- explicit `rebuild` command
- multi-query retrieval and query rewriting
