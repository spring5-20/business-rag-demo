"""LangGraph 在线 RAG：条件路由 + 自我纠错循环

本文件不依赖其它 demo 文件，把检索、重排、提示词、LLM、来源 URL 都写在这里，
方便对照图结构理解每一步。

    START ── chitchat ──→ chitchat ──→ END
          └── retrieve ──→ rerank ──→ generate ── done ──→ END
                                        │ rewrite
                                        └──→ rewrite ──→ retrieve（循环）
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Annotated, Any, TypedDict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/langgraph", tags=["langgraph在线演示"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]


class LangGraphAskRequest(BaseModel):
    """在线问答请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    source_type: str | None = Field(None, description="可选，按来源类型过滤")
    top_k: int = Field(20, ge=1, le=100, description="向量检索候选数")
    rerank_top_n: int = Field(3, ge=1, le=20, description="重排序后保留数")
    max_attempts: int = Field(3, ge=1, le=10, description="自我纠错最多重试次数")


class AskState(TypedDict):
    """在线问答图所有节点共享的状态。"""

    question: str
    source_type: str | None
    top_k: int
    rerank_top_n: int
    max_attempts: int
    attempt: int
    documents: list[Document]
    sources: list[dict[str, Any]]
    answer: str


# 三个提示词模板：回答主链路 / 闲聊 / 检索问题改写
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个知识库问答助手，请只依据以下上下文回答问题，如果上下文没有答案就明确说明不知道。"),
        ("human", "上下文：\n{context}\n\n问题：{question}"),
    ]
)

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的知识库助手，请简洁地回答用户。"),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是检索问题改写助手，把用户问题改写成更适合向量检索的简洁问题，只输出改写后的问题。"),
        ("human", "原问题：{question}\n上次回答：{answer}"),
    ]
)


def build_llm() -> ChatOpenAI:
    """DeepSeek 提供 OpenAI 兼容接口，因此用 ChatOpenAI 调用。"""
    return ChatOpenAI(
        base_url=settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com",
        api_key=SecretStr(settings.DEEPSEEK_API_KEY or ""),
        model=settings.DEEPSEEK_MODEL or "deepseek-chat",
        temperature=0.7,
        streaming=True,
    )


def build_file_url(source_path: str | None) -> str | None:
    """把 OSS object key 转成签名 URL；非 object key 或 OSS 不可用时返回 None。"""
    if not source_path or not looks_like_object_key(source_path):
        return None
    return safe_get_file_url(source_path)


async def embed_question(question: str) -> list[float]:
    """把问题向量化。"""
    return await run_in_threadpool(get_text_embedding, question, EMBEDDING_DIMENSION)


async def search_chunks(
    db: AsyncSession,
    embedding: list[float],
    top_k: int,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """用 pgvector 余弦相似度查询最相关的文本块，可按来源类型过滤。"""
    similarity = 1 - func.cosine_distance(RagChunk.embedding, sa_cast(embedding, Vector(EMBEDDING_DIMENSION)))
    query = (
        select(
            RagChunk.content,
            RagDocument.title,
            RagDocument.source_type,
            RagDocument.source_path,
            similarity.label("similarity"),
        )
        .join(RagDocument, RagChunk.document_id == RagDocument.id)
        .order_by(similarity.desc())
        .limit(top_k)
    )
    if source_type:
        query = query.where(RagDocument.source_type == source_type)
    result = await db.execute(query)
    return [
        {
            "content": row["content"],
            "title": row["title"],
            "source_type": row["source_type"],
            "source_path": row["source_path"],
            "similarity": row["similarity"],
        }
        for row in result.mappings().all()
    ]


async def rerank_chunks(query_text: str, documents: list[Document], top_n: int) -> list[Document]:
    """用 qwen3-rerank 重排序，并把结果映射回 Document。"""
    if not documents:
        return []
    results = await run_in_threadpool(
        rerank_documents,
        query_text,
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
    """把检索到的文本块拼成上下文。"""
    return "\n\n".join(f"【{index}】{document.page_content}" for index, document in enumerate(documents, start=1))


def _is_chitchat(question: str) -> bool:
    """简单规则判断：短句或问候语视为闲聊。"""
    text = question.strip().lower()
    keywords = ("你好", "您好", "hi", "hello", "你是谁", "介绍一下", "早上好", "晚上好", "谢谢")
    return len(text) <= 6 or any(keyword in text for keyword in keywords)


def _decide_route(state: AskState) -> str:
    """路由：闲聊直接回答，其他走检索。"""
    return "chitchat" if _is_chitchat(state["question"]) else "retrieve"


def _after_check(state: AskState) -> str:
    """生成后判断：答案合格或达到重试上限则结束，否则改写重试。"""
    if state["attempt"] >= state["max_attempts"]:
        return "done"
    answer = state.get("answer", "")
    if answer.strip() and "无法回答" not in answer and "不知道" not in answer:
        return "done"
    return "rewrite"


def _documents_to_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """把 Document 列表转成前端可展示的来源信息。"""
    return [
        {
            "content": document.page_content,
            "title": document.metadata.get("title"),
            "source_type": document.metadata.get("source_type"),
            "source_path": document.metadata.get("source_path"),
            "file_url": build_file_url(document.metadata.get("source_path")),
            "similarity": document.metadata.get("similarity"),
            "relevance_score": document.metadata.get("relevance_score"),
        }
        for document in documents
    ]


async def _run_chain_stream(chain: Runnable, inputs: dict[str, Any], config: RunnableConfig) -> str:
    """流式执行 LCEL 链，并拼接出完整文本结果。"""
    parts: list[str] = []
    async for piece in chain.astream(inputs, config=config):
        if piece:
            parts.append(piece)
    return "".join(parts)


def _format_sse(data: dict[str, Any]) -> str:
    """把数据格式化为 SSE 事件字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_rag_graph(db: AsyncSession) -> Any:
    """构建在线 RAG 图：路由 → 检索 → 重排 → 生成 → 自检/改写。"""

    async def retrieve(state: AskState) -> dict[str, Any]:
        """节点：向量检索候选文本块。"""
        embedding = await embed_question(state["question"])
        rows = await search_chunks(db, embedding, state["top_k"], state["source_type"])
        documents = [
            Document(
                page_content=str(row["content"]),
                metadata={
                    "title": row["title"],
                    "source_type": row["source_type"],
                    "source_path": row["source_path"],
                    "similarity": row["similarity"],
                },
            )
            for row in rows
        ]
        return {"documents": documents, "sources": _documents_to_sources(documents)}

    async def rerank(state: AskState) -> dict[str, Any]:
        """节点：重排序。"""
        documents = await rerank_chunks(state["question"], state["documents"], state["rerank_top_n"])
        return {"documents": documents, "sources": _documents_to_sources(documents)}

    async def generate(state: AskState, config: RunnableConfig) -> dict[str, Any]:
        """节点：根据上下文生成回答。"""
        chain: Runnable = PROMPT | build_llm() | StrOutputParser()
        answer = await _run_chain_stream(
            chain,
            {"context": format_docs(state["documents"]), "question": state["question"]},
            config,
        )
        return {"answer": answer}

    async def chitchat(state: AskState, config: RunnableConfig) -> dict[str, Any]:
        """节点：闲聊直接回答。"""
        chain: Runnable = CHITCHAT_PROMPT | build_llm() | StrOutputParser()
        answer = await _run_chain_stream(chain, {"question": state["question"]}, config)
        return {"answer": answer}

    async def rewrite(state: AskState, config: RunnableConfig) -> dict[str, Any]:
        """节点：改写问题，准备重新检索。"""
        chain: Runnable = REWRITE_PROMPT | build_llm() | StrOutputParser()
        new_question = await _run_chain_stream(
            chain,
            {"question": state["question"], "answer": state.get("answer", "")},
            config,
        )
        return {"question": new_question.strip(), "attempt": state["attempt"] + 1}

    graph = StateGraph(AskState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("chitchat", chitchat)
    graph.add_node("rewrite", rewrite)

    graph.add_conditional_edges(START, _decide_route, {"chitchat": "chitchat", "retrieve": "retrieve"})
    graph.add_edge("chitchat", END)
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_conditional_edges("generate", _after_check, {"rewrite": "rewrite", "done": END})
    graph.add_edge("rewrite", "retrieve")
    return graph.compile()


@router.post("/ask")
async def ask(payload: LangGraphAskRequest, db: DbSession) -> StreamingResponse:
    """用 LangGraph 图完成在线问答（条件路由 + 自我纠错），SSE 流式输出。"""
    app = build_rag_graph(db)
    initial_state: AskState = {
        "question": payload.question,
        "source_type": payload.source_type,
        "top_k": payload.top_k,
        "rerank_top_n": payload.rerank_top_n,
        "max_attempts": payload.max_attempts,
        "attempt": 0,
        "documents": [],
        "sources": [],
        "answer": "",
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        """先发来源，再逐 token 发回答，最后发 DONE。"""
        try:
            async for mode, data in app.astream(initial_state, stream_mode=["updates", "messages"]):
                if mode == "updates":
                    for update in data.values():
                        if "sources" in update:
                            yield _format_sse({"sources": update["sources"]})
                elif mode == "messages":
                    chunk, metadata = data
                    if metadata.get("langgraph_node") in ("generate", "chitchat"):
                        content = chunk.content
                        if content:
                            yield _format_sse({"content": content})
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
