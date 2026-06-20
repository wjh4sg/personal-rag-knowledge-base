# Personal RAG Knowledge Base v0.1.1 Design

## Objective

Close the highest-value review gaps in the public v0.1.0 repository without
changing the RAG architecture or persisted data formats.

v0.1.1 adds:

- complete index validation before retrieval;
- an explicit full rebuild command;
- real CLI output examples in the README;
- an embedded architecture diagram;
- an MIT license and package license metadata.

## Scope

### Complete index validation

`search`, `ask`, and `eval` require all three derived/authoritative retrieval
artifacts:

```text
chunks.jsonl
bm25.pkl
non-empty Chroma collection
```

If any artifact is absent or empty, the CLI exits with code `3` and prints:

```text
知识库索引不存在或不完整，请先运行 rag index <文档目录>，
或运行 rag rebuild <文档目录> 全量重建。
```

`stats` remains usable when the index is missing or partial so the user can
inspect the broken state.

### Rebuild command

The new command is:

```text
rag rebuild <docs_path>
```

It performs two phases:

1. Remove only known generated index state:
   - `docs.json`
   - `chunks.jsonl`
   - `bm25.pkl`
   - `stats.json`
   - the configured Chroma directory
2. Invoke the normal indexing pipeline against `docs_path`.

The embedding cache is preserved. Therefore unchanged `chunk_hash` and model
settings produce cache hits during rebuild instead of new embedding requests.

The deletion boundary is deliberately narrow. `rebuild` does not recursively
delete the entire storage base directory and does not remove unrelated files.

If rebuilding fails after cleanup, the command returns the existing provider or
input error code. This behavior is acceptable because `rebuild` is an explicit
destructive recovery operation and can be retried.

### Storage support

Storage adapters expose explicit reset methods:

- manifest stores remove their single known file;
- BM25 removes its pickle and resets in-memory state;
- vector storage deletes the configured persistent Chroma directory and
  recreates a clean client/collection.

The rebuild orchestration lives in the CLI composition layer because it owns all
configured storage paths and then delegates actual indexing to the existing
`Indexer`.

### README demo output

The README gains a “Demo 输出示例” section containing concise, real mock-mode
output for:

- `rag index`
- `rag search`
- `rag ask`
- `rag stats`
- `rag eval`
- `rag rebuild`

Outputs must match current command labels and clearly show:

- Chroma and BM25 completion;
- `source: fusion`;
- a legal `[C1]` citation;
- index status;
- Hit@1, Hit@3, MRR, and citation coverage;
- rebuild cache hits.

### Architecture diagram

Add `docs/architecture.svg`, a repository-native SVG diagram showing:

```text
Documents
  -> scanner/parsers/chunker
  -> embedding cache
  -> Chroma + BM25
  -> RRF
  -> context C1..Cn
  -> generator
  -> citation checker

Evaluation and stats observe the same stores and pipeline.
```

The README embeds the SVG and keeps the text flow as accessible fallback.

### License and version metadata

Add the standard MIT license with copyright:

```text
Copyright (c) 2026 wjh4sg
```

Package metadata changes:

```toml
version = "0.1.1"
license = "MIT"
```

The build-system requirement is raised to a setuptools version verified locally
to accept the SPDX license expression.

Update:

- `CHANGELOG.md`
- package `__version__`
- README version-management section if necessary
- Git tag and GitHub release

## Error Handling

- Missing/partial index: exit `3`, actionable index/rebuild message.
- Missing docs directory for rebuild: exit `2` before deleting existing state.
- Provider failure during rebuild: exit `4`.
- Unexpected filesystem failure: exit `1`, preserving the normal CLI error
  sanitization behavior.

The docs directory must be validated before reset starts. This prevents a typo
in `docs_path` from destroying a currently usable index.

## Testing

Implementation follows red-green-refactor.

Tests cover:

1. `search` returns exit `3` when Chunk and BM25 files exist but Chroma is
   absent/empty.
2. CLI help lists `rebuild`.
3. `rebuild` validates `docs_path` before deleting old state.
4. `rebuild` removes stale manifests, BM25, stats, and Chroma content.
5. `rebuild` preserves the embedding cache and reports cache hits.
6. Unrelated files under the storage base directory survive rebuild.
7. All existing five-command acceptance tests continue to pass.
8. A six-command acceptance test exercises rebuild.
9. Package metadata builds successfully with the MIT SPDX license.
10. Ruff, offline pytest, and the opt-in ModelScope smoke test pass.

## Release

After merged-main verification:

```text
version: 0.1.1
tag: v0.1.1
release: Personal RAG Knowledge Base v0.1.1
```

The release notes summarize:

- safer incomplete-index detection;
- cache-preserving rebuild;
- demo output and architecture diagram;
- MIT licensing.

