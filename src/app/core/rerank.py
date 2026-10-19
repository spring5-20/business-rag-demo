from http import HTTPStatus

import dashscope
from dashscope.api_entities.dashscope_response import ReRankResult

from .config import settings

# 初始化 API Key（只需执行一次）
dashscope.api_key = settings.DASHSCOPE_API_KEY  # type: ignore[assignment]

def rerank_documents(query: str, documents: list[str], top_n: int | None = None) -> list[ReRankResult]:
    """调用 qwen3-rerank 对候选文本重新排序。

    Args:
        query: 检索问题。
        documents: 待重排序的候选文本列表。
        top_n: 返回的最大条数，``None`` 表示返回全部。

    Returns:
        按相关性降序排列的结果列表，每个结果包含 ``index`` 与 ``relevance_score``。
    """
    if not documents:
        return []

    response = dashscope.TextReRank.call(
        model=settings.DASHSCOPE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Rerank 请求失败: {response.code} - {response.message}")
    return list(response.output.results)