# 增量索引

这个系统怎么做增量索引？扫描阶段先计算每个文件的 SHA-256 内容 Hash，并与 docs.json 中的旧状态比较。Hash 不变的文件直接跳过；新增文件执行解析、切分和索引；修改文件先定位并删除旧 Chunk 和旧向量，再写入新版本；已经从目录中消失的文件会从文档状态、Chunk 存储和向量索引中一并删除。

BM25 在 v0.1 中采用全量重建，因为局部更新倒排索引会增加实现复杂度。Chroma 使用 Chunk ID 做定向删除和 upsert。

# Embedding 缓存

Embedding 缓存键由 provider、embedding model、向量维度和 chunk_hash 共同计算。只有文本内容及模型配置都相同时才复用缓存，因此切换模型不会错误使用旧向量。

# 状态一致性

chunks.jsonl 是检索内容的权威记录，Chroma 和 BM25 是可重建的派生索引。JSON、JSONL 和 pickle 文件通过临时文件加原子替换发布，避免中途写入留下半个文件。

