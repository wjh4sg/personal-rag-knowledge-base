# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/)。

## [0.1.2] - 2026-06-21

### Added

- GitHub Actions 在 Python 3.11 和 3.12 上运行 Ruff、离线测试与 wheel 构建。
- README CI 状态徽章与面试讲解重点。

### Changed

- v0.1 面试版进入冻结状态，后续仅接受真实缺陷修复。

## [0.1.1] - 2026-06-21

### Added

- 保留 Embedding 缓存的 `rag rebuild` 全量重建命令。
- README 真实 Demo 输出与 SVG 架构图。
- MIT License 与 SPDX 包元数据。

### Fixed

- `search`、`ask`、`eval` 现在会拒绝缺少 Chroma、BM25 或 Chunk manifest 的不完整索引。
- CLI 命令结束时显式关闭 Chroma client，避免 Windows 文件句柄阻止重建。

## [0.1.0] - 2026-06-21

### Added

- Markdown、TXT 与文本型 PDF 解析。
- Chroma 向量索引与 Jieba BM25 关键词索引。
- Reciprocal Rank Fusion 混合检索。
- 带结构化引用核查的 Mock / ModelScope 问答。
- 基于文件 Hash 的增量索引与模型绑定 Embedding 缓存。
- `index`、`search`、`ask`、`stats`、`eval` 五条 CLI 命令。
- Hit@1、Hit@3、MRR 与引用覆盖率评估。
- 离线测试套件和 ModelScope 在线烟雾测试。

[0.1.2]: https://github.com/wjh4sg/personal-rag-knowledge-base/releases/tag/v0.1.2
[0.1.1]: https://github.com/wjh4sg/personal-rag-knowledge-base/releases/tag/v0.1.1
[0.1.0]: https://github.com/wjh4sg/personal-rag-knowledge-base/releases/tag/v0.1.0

