# Personal RAG Knowledge Base

一个面向面试展示的本地多源文档 RAG MVP。它支持 Markdown、TXT 和文本型
PDF，提供增量索引、Chroma 向量检索、BM25 关键词检索、RRF 融合、带引用问答
以及 Hit@K / MRR 评估。

## 安装

需要 Python 3.11 或更高版本，推荐 Python 3.12：

```powershell
py -3.12 -m pip install -e ".[dev]"
```

安装后可使用 `rag` 命令。如果 Python Scripts 目录不在 `PATH`，可使用：

```powershell
py -3.12 -m personal_rag.cli --help
```

## 五条演示命令

```powershell
rag index ./examples/docs
rag search "RAG 为什么需要 Rerank？"
rag ask "这个系统怎么做增量索引？"
rag stats
rag eval ./eval/dataset.json
```

默认配置使用确定性 Mock embedding 和 Mock generator，因此断网也能展示完整
链路。第二次执行 `rag index` 会跳过所有未变化文件。

## 使用魔搭 API

在 Windows 用户环境变量中创建：

```text
MODELSCOPE_API_KEY=<你的魔搭 SDK Token>
```

不要把 Token 写进 YAML、源码或 Git。随后把
[`config/config.yaml`](config/config.yaml) 中的 provider mode 改为：

```yaml
provider:
  mode: api
```

默认真实模型：

```text
Embedding: Qwen/Qwen3-Embedding-0.6B
LLM: Qwen/Qwen3-30B-A3B-Instruct-2507
Base URL: https://api-inference.modelscope.cn/v1
```

Embedding 请求显式发送 `encoding_format: float`。魔搭免费 API 用于试用和评估，
不保证生产 SLA；API 问答失败且 `fallback_to_mock: true` 时，`ask` 会明确显示
`mock-fallback`。

## 架构

```text
文档目录
  -> 文件 Hash 与增量扫描
  -> Markdown / TXT / PDF 解析
  -> Document / Chunk
  -> Embedding 缓存
  -> Chroma + BM25
  -> Vector / BM25 两路召回
  -> RRF 融合
  -> C1..Cn 上下文
  -> Mock 或 ModelScope LLM
  -> 引用合法性检查
```

`chunks.jsonl` 是检索内容的权威记录。Chroma 与 BM25 是派生索引。引用检查保证
模型只能引用本轮检索到的 Chunk，但它不等价于逐句事实忠实性评估。

## 数据与配置

- 默认配置：`config/config.yaml`
- 文档状态：`data/storage/docs.json`
- Chunk：`data/storage/chunks.jsonl`
- BM25：`data/storage/bm25.pkl`
- Chroma：`data/storage/chroma/`
- Embedding 缓存：`data/cache/embeddings/`

相对源路径统一保存为 POSIX 格式，不写入绝对文档路径。缓存键绑定 provider、
模型、维度与 `chunk_hash`。

## 测试

离线测试：

```powershell
py -3.12 -m pytest -m "not live" -q
```

魔搭在线烟雾测试：

```powershell
py -3.12 -m pytest tests/live/test_modelscope.py -m live -q
```

静态检查：

```powershell
py -3.12 -m ruff check .
```

## 常见错误

- `401/403`：确认用户环境变量 `MODELSCOPE_API_KEY` 已设置，并重启终端或 Codex。
- `429`：免费额度或频率限制已触发，稍后重试，或切回 `mock`。
- 请求超时：检查网络并调整 `provider.timeout_seconds`。
- 提示索引不存在：先运行 `rag index ./examples/docs`。
- Chroma/BM25 损坏：删除 `data/storage` 后重新执行 `rag index`。

## v0.1 边界

第一版不支持扫描 PDF OCR、复杂表格 PDF、图片、多用户、Web UI、Agent、多轮长期
记忆、真实 cross-encoder reranker 或 RAGAS。它们属于后续增强项。

