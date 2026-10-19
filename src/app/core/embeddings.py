from http import HTTPStatus
from typing import cast

import dashscope

from .config import settings

# 向量维度，需与 rag_chunks.embedding 的 vector 列维度一致
EMBEDDING_DIMENSION = 1536

# 初始化 API Key（只需执行一次）
dashscope.api_key = settings.DASHSCOPE_API_KEY  # type: ignore[assignment]


def get_text_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """
    调用通义千问文本嵌入模型，返回文本的密集向量和稀疏向量。

    Args:
        text (str): 输入文本
        dimension (int): 向量维度，默认 EMBEDDING_DIMENSION（需与 documents 表的 vector 列维度一致）
    """
    resp = dashscope.TextEmbedding.call(
        model=settings.DASHSCOPE_MODEL,  # type: ignore[arg-type]
        input=text,
        dimension=dimension,
        output_type="dense&sparse",
    )

    if resp.status_code == HTTPStatus.OK:
        emb = resp.output["embeddings"][0]
        return cast(list[float], emb["embedding"])  # 密集向量列表

    else:
        raise RuntimeError(f"Embedding 请求失败: {resp.code} - {resp.message}")
