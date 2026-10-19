"""LangGraph 在线 RAG：人机协作审批（human-in-the-loop）。

与 ``langgraph_ask.py`` 的自动重试版对照，这里把「自动重试」升级成
「自动判断 + 人工审批」的混合模式：

    START ── chitchat ──→ chitchat ──→ END
          └── retrieve ──→ rerank ──→ generate ──合格──→ END
                                          │
                                          └──不合格──→ approval ⏸ ── retry ──→ rewrite ──→ retrieve
                                                              └── accept ──→ END

答案合格（检索到了内容、模型也正常作答）时直接返回，不打扰人；
只有答案疑似不合格（没检索到内容、空答案、或模型说「不知道」）时才暂停，
由人决定是重写问题再试，还是接受这个答案。

实现方式（langgraph 0.2.x 的静态中断点）：

1. 编译时传入 ``checkpointer`` 保存每一步的图状态；
2. ``interrupt_before=["approval"]`` 让图在进入 approval 节点前暂停；
3. 暂停后人工读取待审内容，用 ``aupdate_state`` 写入决定（approved / 新问题）；
4. 再以 ``astream(None, config)`` 恢复执行，approval 节点之后的边按决定路由。

状态存在进程内的 MemorySaver 里，服务重启即丢，仅供本地测试。
"""

import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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

router = APIRouter(prefix="/langgraph", tags=["langgraph人机协作审批"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]

# 图状态检查点保存器：进程内内存实现，服务重启后未审批完的会话会丢失。
# 注意它必须在所有请求间共享，审批暂停与恢复才能命中同一个 thread_id 的状态。
CHECKPOINTER = MemorySaver()

# 前端测试页面（单文件 HTML，与接口同源，无需处理 CORS）。
UI_FILE = Path(__file__).resolve().parents[2] / "static" / "langgraph_approval.html"


class ApprovalAskRequest(BaseModel):
    """发起一次带人工审批的在线问答。"""

    question: str = Field(..., min_length=1, description="用户问题")
    source_type: str | None = Field(None, description="可选，按来源类型过滤")
    top_k: int = Field(20, ge=1, le=100, description="向量检索候选数")
    rerank_top_n: int = Field(3, ge=1, le=20, description="重排序后保留数")
    max_attempts: int = Field(3, ge=1, le=10, description="人工最多重试次数")


class ApprovalResumeRequest(BaseModel):
    """提交人工审批决定，恢复被暂停的问答图。"""

    thread_id: str = Field(..., min_length=1, description="发起提问时返回的会话 id")
    action: Literal["accept", "retry"] = Field(..., description="accept=接受答案并结束；retry=改写问题重新检索")
    question: str | None = Field(None, description="retry 时可选：人工指定新的检索问题")


class ApprovalState(TypedDict):
    """审批问答图所有节点共享的状态。

    注意：图带了 checkpointer，而 langgraph 0.2.x 在快照里会丢掉值为 ``None`` 的
    channel（恢复后该 key 直接不存在）。所以这里的可选过滤条件用空字符串表示
    “不过滤”，不能用 None，否则重试循环读 state["source_type"] 会 KeyError。
    """

    question: str
    source_type: str
    top_k: int
    rerank_top_n: int
    max_attempts: int
    attempt: int
    documents: list[Document]
    sources: list[dict[str, Any]]
    answer: str
    approved: bool


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
    source_type: str = "",
) -> list[dict[str, Any]]:
    """用 pgvector 余弦相似度查询最相关的文本块；source_type 为空串表示不过滤。"""
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


def _decide_route(state: ApprovalState) -> str:
    """路由：闲聊直接回答，其他走检索。"""
    return "chitchat" if _is_chitchat(state["question"]) else "retrieve"


def _check_answer(state: ApprovalState) -> str:
    """生成后自动判断：合格直接结束，疑似不合格才进入人工审批。

    用简单启发式代替 LLM 打分：没检索到内容、答案为空、或答案里出现
    「不知道 / 无法回答」等措辞，都视为可能没答好，交给人工决定。
    达到最大重试次数后不再打扰人，直接结束（与自动重试版行为一致）。
    """
    if state.get("attempt", 0) >= state.get("max_attempts", 0):
        return "done"
    answer = state.get("answer", "")
    if not state.get("documents") or not answer.strip():
        return "approval"
    unsure_tokens = ("不知道", "无法回答", "没有相关", "未提及", "无法确定")
    if any(token in answer for token in unsure_tokens):
        return "approval"
    return "done"


def _route_after_approval(state: ApprovalState) -> str:
    """审批后路由：接受则结束；拒绝且未达上限则改写问题重试。"""
    if state.get("approved"):
        return "done"
    if state.get("attempt", 0) >= state.get("max_attempts", 0):
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


def _approval_payload(values: dict[str, Any]) -> dict[str, Any]:
    """把暂停时的图状态整理成前端审批面板需要的内容。"""
    return {
        "approval_required": True,
        "question": values.get("question"),
        "answer": values.get("answer"),
        "sources": values.get("sources", []),
        "attempt": values.get("attempt", 0),
        "max_attempts": values.get("max_attempts", 0),
    }


def build_approval_graph(db: AsyncSession) -> Any:
    """构建带人工审批断点的在线 RAG 图。

    ``interrupt_before=["approval"]`` 让图每次到达 approval 节点前暂停，
    等待人工通过 ``aupdate_state`` 写入 ``approved`` 后再恢复。
    """

    async def retrieve(state: ApprovalState) -> dict[str, Any]:
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

    async def rerank(state: ApprovalState) -> dict[str, Any]:
        """节点：重排序。"""
        documents = await rerank_chunks(state["question"], state["documents"], state["rerank_top_n"])
        return {"documents": documents, "sources": _documents_to_sources(documents)}

    async def generate(state: ApprovalState, config: RunnableConfig) -> dict[str, Any]:
        """节点：根据上下文生成回答。"""
        chain: Runnable = PROMPT | build_llm() | StrOutputParser()
        answer = await _run_chain_stream(
            chain,
            {"context": format_docs(state["documents"]), "question": state["question"]},
            config,
        )
        return {"answer": answer}

    async def chitchat(state: ApprovalState, config: RunnableConfig) -> dict[str, Any]:
        """节点：闲聊直接回答（不检索，也不进入审批）。"""
        chain: Runnable = CHITCHAT_PROMPT | build_llm() | StrOutputParser()
        answer = await _run_chain_stream(chain, {"question": state["question"]}, config)
        return {"answer": answer}

    async def rewrite(state: ApprovalState, config: RunnableConfig) -> dict[str, Any]:
        """节点：改写问题，准备重新检索。"""
        chain: Runnable = REWRITE_PROMPT | build_llm() | StrOutputParser()
        new_question = await _run_chain_stream(
            chain,
            {"question": state["question"], "answer": state.get("answer", "")},
            config,
        )
        return {"question": new_question.strip(), "attempt": state["attempt"] + 1}

    async def approval(state: ApprovalState) -> dict[str, Any]:
        """节点：人工审批占位（只有答案疑似不合格时才会走到这里）。

        图在进入本节点前就会暂停（interrupt_before），人工通过 aupdate_state 写入
        approved / question 后才会执行到这里，真正的分流交给后面的条件边。

        注意：LangGraph 要求节点至少写一个状态字段，返回空字典会抛 InvalidUpdateError，
        所以这里把人工写入的 approved 原样回写一次（不做任何判断）。
        """
        return {"approved": state["approved"]}

    graph = StateGraph(ApprovalState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("chitchat", chitchat)
    graph.add_node("rewrite", rewrite)
    graph.add_node("approval", approval)

    graph.add_conditional_edges(START, _decide_route, {"chitchat": "chitchat", "retrieve": "retrieve"})
    graph.add_edge("chitchat", END)
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_conditional_edges("generate", _check_answer, {"approval": "approval", "done": END})
    graph.add_conditional_edges("approval", _route_after_approval, {"rewrite": "rewrite", "done": END})
    graph.add_edge("rewrite", "retrieve")
    return graph.compile(checkpointer=CHECKPOINTER, interrupt_before=["approval"])


async def _stream_graph(
    app: Any,
    graph_input: Any,
    config: RunnableConfig,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """把图的一次执行流转成 SSE 事件流。

    - ``thread_id``：首帧下发，前端后续审批恢复要用；
    - ``sources``：检索/重排完成时下发来源；
    - ``content``：generate / chitchat 逐 token 下发回答；
    - ``approval_required``：图暂停在审批断点时下发待审内容。
    """
    try:
        yield _format_sse({"thread_id": thread_id})
        async for mode, data in app.astream(graph_input, config, stream_mode=["updates", "messages"]):
            if mode == "updates":
                if "__interrupt__" in data:
                    snapshot = await app.aget_state(config)
                    yield _format_sse(_approval_payload(dict(snapshot.values)))
                    continue
                for update in data.values():
                    if isinstance(update, dict) and "sources" in update:
                        yield _format_sse({"sources": update["sources"]})
            elif mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") in ("generate", "chitchat"):
                    content = chunk.content
                    if content:
                        yield _format_sse({"content": content})
    except Exception as exc:
        logger.exception("审批问答流式调用失败")
        yield _format_sse({"error": str(exc)})
    finally:
        yield "data: [DONE]\n\n"


def _sse_response(event_stream: AsyncGenerator[str, None]) -> StreamingResponse:
    """统一包装 SSE 响应头。"""
    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/approval/ask")
async def approval_ask(payload: ApprovalAskRequest, db: DbSession) -> StreamingResponse:
    """发起问答：流式返回答案；只有答案疑似不合格时才暂停等待人工审批。"""
    thread_id = uuid4().hex
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    app = build_approval_graph(db)
    initial_state: ApprovalState = {
        "question": payload.question,
        "source_type": payload.source_type or "",
        "top_k": payload.top_k,
        "rerank_top_n": payload.rerank_top_n,
        "max_attempts": payload.max_attempts,
        "attempt": 0,
        "documents": [],
        "sources": [],
        "answer": "",
        "approved": False,
    }
    return _sse_response(_stream_graph(app, initial_state, config, thread_id))


@router.post("/approval/resume")
async def approval_resume(payload: ApprovalResumeRequest, db: DbSession) -> StreamingResponse:
    """提交审批决定：accept 接受当前答案并结束，retry 改写问题后重新检索生成。"""
    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}
    app = build_approval_graph(db)
    snapshot = await app.aget_state(config)
    if not snapshot.values or "approval" not in (snapshot.next or ()):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在或当前不处于审批暂停状态（服务重启后内存状态会丢失）",
        )

    values = dict(snapshot.values)
    if payload.action == "retry" and values.get("attempt", 0) >= values.get("max_attempts", 0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="已达到最大重试次数，无法继续重试",
        )

    updates: dict[str, Any] = {"approved": payload.action == "accept"}
    if payload.action == "retry" and payload.question:
        updates["question"] = payload.question
    await app.aupdate_state(config, updates)
    return _sse_response(_stream_graph(app, None, config, payload.thread_id))


@router.get("/approval/ui", include_in_schema=False)
async def approval_ui() -> FileResponse:
    """返回人机协作审批的测试页面（单文件 HTML）。"""
    if not UI_FILE.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="测试页面文件不存在")
    return FileResponse(UI_FILE, media_type="text/html")
