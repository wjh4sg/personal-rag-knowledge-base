from __future__ import annotations

from dataclasses import dataclass

from personal_rag.core.schema import RetrievedChunk


@dataclass(frozen=True)
class BuiltContext:
    prompt: str
    citation_map: dict[str, RetrievedChunk]


def build_context(question: str, chunks: list[RetrievedChunk]) -> BuiltContext:
    citation_map: dict[str, RetrievedChunk] = {}
    parts = [
        "你是一个严谨的个人知识库问答助手。",
        "只能基于给定资料回答问题。",
        "如果资料不足，请回答：当前知识库信息不足，无法确定。",
        "回答中必须使用 [C1]、[C2] 这样的引用编号。",
        "",
        f"用户问题：{question}",
        "",
        "可用资料：",
    ]
    for index, chunk in enumerate(chunks, start=1):
        citation_id = f"C{index}"
        citation_map[citation_id] = chunk
        parts.extend(
            [
                "",
                (
                    f"[{citation_id}] 来源：{chunk.metadata.get('source_path', '')} "
                    f"页码：{chunk.metadata.get('page') or ''} "
                    f"标题：{chunk.metadata.get('heading') or ''}"
                ),
                chunk.text,
            ]
        )
    parts.extend(["", "请输出：", "回答：", "引用："])
    return BuiltContext(prompt="\n".join(parts), citation_map=citation_map)

