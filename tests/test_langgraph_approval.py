"""人机协作审批接口的离线测试。

这些用例只覆盖不需要数据库 / LLM 的部分：测试页面、会话校验、路由判断与图编译。
真正的检索与生成需要 PostgreSQL、DashScope、DeepSeek，不在单测范围内。
"""

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.app.api.v1.langgraph_approval import (
    ApprovalState,
    _check_answer,
    _route_after_approval,
    build_approval_graph,
)
from src.app.main import app


def _base_state(**overrides: object) -> ApprovalState:
    """构造一份最小的图状态，供纯函数测试使用。"""
    state: ApprovalState = {
        "question": "这份文档讲了什么？",
        "source_type": "",
        "top_k": 20,
        "rerank_top_n": 3,
        "max_attempts": 3,
        "attempt": 0,
        "documents": [],
        "sources": [],
        "answer": "",
        "approved": False,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_check_answer_passes_relevant_answer() -> None:
    """检索到内容且答案正常时，应直接结束、不进入人工审批。"""
    state = _base_state(documents=[object()], answer="这个项目使用了 LangChain 做文档切分。")
    assert _check_answer(state) == "done"


def test_check_answer_flags_empty_retrieval() -> None:
    """没检索到任何内容时，答案多半不可信，应进入人工审批。"""
    assert _check_answer(_base_state(documents=[], answer="随便答点什么")) == "approval"


def test_check_answer_flags_unsure_answer() -> None:
    """模型明说「不知道」时应进入人工审批。"""
    assert _check_answer(_base_state(documents=[object()], answer="我不知道，上下文里没有相关信息。")) == "approval"


def test_check_answer_stops_at_max_attempts() -> None:
    """达到最大重试次数后不再打扰人，直接结束。"""
    state = _base_state(documents=[], answer="", attempt=3, max_attempts=3)
    assert _check_answer(state) == "done"


def test_route_after_approval_accept_ends() -> None:
    """人工接受时应结束，而不是继续重试。"""
    assert _route_after_approval(_base_state(approved=True)) == "done"


def test_route_after_approval_retry_when_rejected() -> None:
    """人工拒绝且未达上限时应进入改写重试。"""
    assert _route_after_approval(_base_state(approved=False, attempt=1, max_attempts=3)) == "rewrite"


def test_route_after_approval_stops_at_max_attempts() -> None:
    """达到最大重试次数后即使人工点重试也必须结束。"""
    assert _route_after_approval(_base_state(approved=False, attempt=3, max_attempts=3)) == "done"


def test_build_approval_graph_contains_approval_node() -> None:
    """图应以 approval 节点作为人工中断点，并带上 checkpointer。"""
    graph = build_approval_graph(None)  # type: ignore[arg-type]  # 只编译，不执行节点

    assert "approval" in set(graph.get_graph().nodes)
    assert graph.checkpointer is not None


@pytest.mark.anyio
async def test_approval_ui_served() -> None:
    """测试页面应能以 HTML 正常返回。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/langgraph/approval/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "人机协作审批" in response.text


def _patch_offline_deps(monkeypatch: pytest.MonkeyPatch, llm_response: str) -> None:
    """用桩替换检索与 LLM，让图能在没有数据库/外部模型的情况下跑完。"""
    from src.app.api.v1 import langgraph_approval as approval

    async def fake_search_chunks(db: object, embedding: list[float], top_k: int, source_type: str = "") -> list[dict]:
        return [
            {
                "content": "片段内容",
                "title": "演示文档",
                "source_type": "pdf",
                "source_path": "demo.pdf",
                "similarity": 0.91,
            }
        ]

    async def fake_rerank_chunks(query: str, documents: list, top_n: int) -> list:
        return documents

    async def fake_embed_question(question: str) -> list[float]:
        return [0.0] * 1536

    monkeypatch.setattr(approval, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(approval, "rerank_chunks", fake_rerank_chunks)
    monkeypatch.setattr(approval, "embed_question", fake_embed_question)
    monkeypatch.setattr(approval, "build_llm", lambda: FakeListChatModel(responses=[llm_response]))


@pytest.mark.anyio
async def test_relevant_answer_returns_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """答案正常时应直接返回，不暂停、不打扰人。"""
    _patch_offline_deps(monkeypatch, "这个项目使用了 LangChain 做文档切分。")
    from src.app.api.v1 import langgraph_approval as approval

    graph = approval.build_approval_graph(None)  # type: ignore[arg-type]
    config = {"configurable": {"thread_id": "test-good-answer"}}

    async for _ in graph.astream(_base_state(), config, stream_mode=["updates"]):
        pass

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()  # 已结束，没有停在审批点
    assert snapshot.values["answer"] == "这个项目使用了 LangChain 做文档切分。"
    assert len(snapshot.values["sources"]) == 1


@pytest.mark.anyio
async def test_unsure_answer_pauses_then_retry_then_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    """答案疑似不合格时才暂停：拒绝重试 → 再次暂停 → 接受结束。

    覆盖两个曾经踩过的坑：审批节点不能返回空字典、状态里的可选过滤条件不能用 None。
    """
    _patch_offline_deps(monkeypatch, "我不知道，上下文里没有相关信息。")
    from src.app.api.v1 import langgraph_approval as approval

    graph = approval.build_approval_graph(None)  # type: ignore[arg-type]
    config = {"configurable": {"thread_id": "test-unsure-answer"}}

    # 1) 首次执行：答案不合格，暂停在审批点
    async for _ in graph.astream(_base_state(), config, stream_mode=["updates"]):
        pass
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("approval",)
    assert snapshot.values["answer"] == "我不知道，上下文里没有相关信息。"
    assert len(snapshot.values["sources"]) == 1

    # 2) 人工拒绝：改写问题后重新检索，再次暂停
    await graph.aupdate_state(config, {"approved": False, "question": "文档主题是什么？"})
    async for _ in graph.astream(None, config, stream_mode=["updates"]):
        pass
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("approval",)
    assert snapshot.values["attempt"] == 1

    # 3) 人工接受：走到 END
    await graph.aupdate_state(config, {"approved": True})
    async for _ in graph.astream(None, config, stream_mode=["updates"]):
        pass
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()


@pytest.mark.anyio
async def test_resume_unknown_thread_returns_404() -> None:
    """对不存在的会话提交审批应返回 404，而不是新建一个空图状态。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/langgraph/approval/resume",
            json={"thread_id": "not-exist", "action": "accept"},
        )

    assert response.status_code == 404
