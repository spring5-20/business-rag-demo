"""在线 PDF 问答接口。

流程：问题向量化 → pgvector 检索 top_k 候选块 → qwen rerank 重排到 top_n
      → 组装上下文 → DeepSeek（OpenAI 兼容）LCEL 流式回答 + 返回引用来源。
"""

import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import cast as sa_cast
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db.database import async_get_db
from ...core.db.models import RagChunk, RagDocument, Vector
from ...core.embeddings import EMBEDDING_DIMENSION, get_text_embedding
from ...core.oss import looks_like_object_key, safe_get_file_url
from ...core.rerank import rerank_documents
from ...core.utils import cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo", tags=["pdf在线查询"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]

# 检索+rerank 结果在 Redis 的缓存前缀与过期秒数。
# 缓存 key 需带 metadata 里的 source_path，故用 v2 命名空间一次性作废旧缓存。
RERANK_CACHE_PREFIX = "rerank:v2"
RERANK_CACHE_TTL = 3600


class SearchRequest(BaseModel):
    """在线问答请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(20, ge=1, le=100, description="向量检索返回的候选文本块数量")
    rerank_top_n: int = Field(3, ge=1, le=20, description="qwen3-rerank 重排序后保留的文本块数量")


async def embed_question(question: str) -> list[float]:
    """把问题向量化。"""
    return await run_in_threadpool(get_text_embedding, question, EMBEDDING_DIMENSION)


async def search_chunks(db: AsyncSession, embedding: list[float], top_k: int) -> list[dict[str, float | str | None]]:
    """用 pgvector 余弦相似度查询最相关的文本块。"""
    similarity = 1 - func.cosine_distance(RagChunk.embedding, sa_cast(embedding, Vector(EMBEDDING_DIMENSION)))
    query = (
        select(RagChunk.content, RagDocument.title, RagDocument.source_path, similarity.label("similarity"))
        .join(RagDocument, RagChunk.document_id == RagDocument.id)
        .order_by(similarity.desc())
        .limit(top_k)
    )
    result = await db.execute(query)
    return [
        {
            "content": row["content"],
            "title": row["title"],
            "source_path": row["source_path"],
            "similarity": round(float(row["similarity"]), 4),
        }
        for row in result.mappings().all()
    ]


async def rerank_chunks(query: str, documents: list[Document], top_n: int) -> list[Document]:
    """调用 rerank 模型重排，并把结果映射回 Document（附上 relevance_score）。"""
    if not documents:
        return []

    results = await run_in_threadpool(
        rerank_documents,
        query,
        [document.page_content for document in documents],
        top_n,
    )
    reranked: list[Document] = []
    for result in results:
        document = documents[result.index]
        document.metadata["relevance_score"] = result.relevance_score
        reranked.append(document)
    return reranked


def format_docs(documents: list[Document]) -> str:
    """把重排序后的 Document 列表拼成一段上下文。"""
    return "\n\n".join(f"【{index}】{document.page_content}" for index, document in enumerate(documents, start=1))


def _query_hash(question: str) -> str:
    """对问题做稳定哈希（sha256 截断 16 位），用作缓存 key 的一段。"""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


async def _kb_version_ms(db: AsyncSession) -> int:
    """知识库版本号：rag_documents 最新 created_at 的毫秒时间戳，每次请求查一次。"""
    result = await db.execute(select(func.max(RagDocument.created_at)))
    latest = result.scalar_one_or_none()
    if latest is None:  # 空库
        return 0
    if latest.tzinfo is None:  # 兜底：naive datetime 一律按 UTC 解释
        latest = latest.replace(tzinfo=UTC)
    return int(latest.timestamp() * 1000)


def _rerank_cache_key(kb_version: int, question: str, top_k: int, top_n: int) -> str:
    """生成检索+rerank 整条链结果的缓存 key。"""
    return f"{RERANK_CACHE_PREFIX}:{kb_version}:{_query_hash(question)}:{top_k}:{top_n}"


def _documents_to_json(documents: list[Document]) -> str:
    """把 list[Document] 序列化为 JSON 字符串（只保留 page_content 与 metadata）。"""
    return json.dumps(
        [{"page_content": document.page_content, "metadata": document.metadata} for document in documents],
        ensure_ascii=False,
    )


def _documents_from_json(raw: str) -> list[Document]:
    """把 JSON 字符串反序列化为 list[Document]，完整还原 metadata。"""
    data = json.loads(raw)
    return [
        Document(page_content=item["page_content"], metadata=dict(item["metadata"]))
        for item in data
    ]


async def _retrieve_and_rerank(db: AsyncSession, question: str, top_k: int, top_n: int) -> list[Document]:
    """向量检索 top_k 候选块 → qwen rerank 到 top_n，返回最终 Document 列表。"""
    embedding = await embed_question(question)
    rows = await search_chunks(db, embedding, top_k)
    documents = [
        Document(
            page_content=str(row["content"]),
            metadata={
                "title": row["title"],
                "similarity": row["similarity"],
                "source_path": row["source_path"],
            },
        )
        for row in rows
    ]
    return await rerank_chunks(question, documents, top_n)


async def _get_reranked_documents(
    db: AsyncSession, question: str, top_k: int, top_n: int
) -> tuple[list[Document], bool]:
    """带 Redis 缓存的整条链取值。

    命中：返回反序列化后的 top_n Document（不检索、不 rerank），``from_cache=True``。
    未命中：现算并写入 Redis（带 TTL），``from_cache=False``。
    Redis 不可用或出错时优雅降级：每次现算、不缓存、不抛异常。
    """
    if not cache.is_available():
        documents = await _retrieve_and_rerank(db, question, top_k, top_n)
        return documents, False

    kb_version = await _kb_version_ms(db)
    key = _rerank_cache_key(kb_version, question, top_k, top_n)
    raw = await cache.get_value(key)
    if raw is not None:
        logger.debug("Rerank 缓存命中: %s", key)
        return _documents_from_json(raw), True

    logger.debug("Rerank 缓存未命中: %s", key)
    documents = await _retrieve_and_rerank(db, question, top_k, top_n)
    if await cache.set_value(key, _documents_to_json(documents), RERANK_CACHE_TTL):
        logger.debug("写入 Rerank 缓存: %s（TTL %s 秒）", key, RERANK_CACHE_TTL)
    return documents, False


def build_llm() -> ChatOpenAI:
    """DeepSeek 提供 OpenAI 兼容接口，因此直接用 ChatOpenAI 调用。"""
    return ChatOpenAI(
        base_url=settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com",
        api_key=SecretStr(settings.DEEPSEEK_API_KEY or ""),
        model=settings.DEEPSEEK_MODEL or "deepseek-chat",
        temperature=0.7,
        streaming=True,
    )


# LCEL 使用的提示词模板
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个知识库问答助手，请只依据以下上下文回答问题，如果上下文没有答案就明确说明不知道。"),
        ("human", "上下文：\n{context}\n\n问题：{question}"),
    ]
)


def _format_sse(data: dict[str, Any]) -> str:
    """把数据格式化为 SSE 事件字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """从重排后的 Document 列表生成 sources，并附上 OSS 临时签名 URL。

    source_path 是 OSS object key 时，每次现生成签名 URL（有时效，不落缓存）；
    旧数据（本地路径）或 OSS 不可用时 ``url`` 为 None，供前端判空隐藏入口。
    """
    sources: list[dict[str, Any]] = []
    for document in documents:
        source_path = document.metadata.get("source_path")
        url = safe_get_file_url(source_path) if looks_like_object_key(source_path) else None
        sources.append(
            {
                "content": document.page_content,
                "title": document.metadata["title"],
                "similarity": document.metadata.get("similarity"),
                "relevance_score": document.metadata.get("relevance_score"),
                "url": url,
            }
        )
    return sources


@router.post("/ask")
async def ask(payload: SearchRequest, db: DbSession) -> StreamingResponse:
    """在线问答（SSE 流式）：检索 → qwen3-rerank 重排 → LCEL 流式回答。

    检索+rerank 的最终结果按 ``rerank:v2:版本:问题哈希:top_k:top_n`` 缓存到 Redis，
    命中时零检索、零 rerank；未命中才现算并回写。LCEL 链只负责把已拼好的
    context 喂给 LLM 流式回答，链内不再重复检索/重排。
    """
    documents, _from_cache = await _get_reranked_documents(
        db, payload.question, payload.top_k, payload.rerank_top_n
    )
    sources = _build_sources(documents)
    # 用同一份 documents 预拼 context，LCEL 链不再含检索/重排段
    context = format_docs(documents)
    chain: Any = PROMPT | build_llm() | StrOutputParser()
    inputs: dict[str, str] = {"context": context, "question": payload.question}

    async def event_stream() -> AsyncGenerator[str, None]:
        """逐块产出 SSE 事件：先发来源，再流式发回答，最后发 DONE。"""
        yield _format_sse({"sources": sources})
        try:
            async for piece in chain.astream(inputs):
                if piece:
                    yield _format_sse({"content": piece})
        except Exception as exc:
            logger.exception("在线问答流式调用失败")
            yield _format_sse({"error": str(exc)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
