"""LangGraph 入门：离线 PDF 入库（线性图）。

把离线流程的四步画成一张 StateGraph：

    START → load → split → embed → save → END

本文件刻意保持简单，用来建立 State / Node / Edge / Graph 的基础概念。
"""

from pathlib import Path
from typing import Annotated, Any, TypedDict

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db.database import async_get_db
from ...core.db.models import RagChunk, RagDocument
from ...core.embeddings import EMBEDDING_DIMENSION, get_text_embedding
from ...core.oss import upload_and_get_key

router = APIRouter(prefix="/langgraph", tags=["langgraph离线演示"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]

# 中文友好的分隔符优先级：先按段落、换行，再按中文标点，最后按空格与单个字符兜底
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class LangGraphPdfRequest(BaseModel):
    """离线入库请求。"""

    path: str = Field(..., min_length=1, description="PDF 相对 DOCUMENT_DIR 的路径")


class LangGraphPdfResponse(BaseModel):
    """离线入库结果。"""

    filename: str
    pages: int
    total_chars: int
    chunk_count: int
    document_id: int
    previews: list[dict[str, int | str]] = Field(default_factory=list)


class IngestState(TypedDict):
    """离线入库图所有节点共享的状态。"""

    path: str
    object_key: str
    documents: list[Document]
    chunks: list[Document]
    embeddings: list[list[float]]
    document_id: int
    pages: int
    total_chars: int


def _resolve_pdf_path(path: str) -> Path:
    """把相对路径解析到 DOCUMENT_DIR 内，防止目录穿越。"""
    root = settings.DOCUMENT_DIR.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="路径必须在 DOCUMENT_DIR 目录内",
        )
    return candidate


def load_pdf(path: Path) -> list[Document]:
    """用 PyPDFLoader 加载 PDF，每一页是一条 Document。"""
    return PyPDFLoader(str(path)).load()


def build_ingest_graph(db: AsyncSession, chunk_size: int, chunk_overlap: int) -> Any:
    """把「加载 → 切分 → 向量化 → 入库」组装成一张编译后的线性图。

    LangGraph 节点函数只接收 state，因此请求级的 db / chunk_size / chunk_overlap
    通过闭包捕获，每个请求独立构图、编译一次再运行一次。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )

    async def load(state: IngestState) -> dict[str, Any]:
        """节点 1：加载 PDF。"""
        documents = await run_in_threadpool(load_pdf, Path(state["path"]))
        if not documents or not "".join(document.page_content for document in documents).strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="未能从 PDF 中提取到文本",
            )
        return {
            "documents": documents,
            "pages": len(documents),
            "total_chars": sum(len(document.page_content) for document in documents),
        }

    def split(state: IngestState) -> dict[str, Any]:
        """节点 2：切分。"""
        return {"chunks": splitter.split_documents(state["documents"])}

    async def embed(state: IngestState) -> dict[str, Any]:
        """节点 3：向量化。"""
        embeddings = [
            await run_in_threadpool(get_text_embedding, chunk.page_content, EMBEDDING_DIMENSION)
            for chunk in state["chunks"]
        ]
        return {"embeddings": embeddings}

    async def save(state: IngestState) -> dict[str, Any]:
        """节点 4：入库。"""
        path = Path(state["path"])
        document = RagDocument(title=path.stem or path.name, source_type="pdf", source_path=state["object_key"])
        db.add(document)
        await db.flush()
        for index, (chunk, embedding) in enumerate(zip(state["chunks"], state["embeddings"], strict=True)):
            db.add(
                RagChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk.page_content,
                    embedding=embedding,
                    token_count=len(chunk.page_content),
                )
            )
        await db.commit()
        return {"document_id": document.id}

    graph = StateGraph(IngestState)
    graph.add_node("load", load)
    graph.add_node("split", split)
    graph.add_node("embed", embed)
    graph.add_node("save", save)
    graph.add_edge(START, "load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "save")
    graph.add_edge("save", END)
    return graph.compile()


@router.post("/pdf", response_model=LangGraphPdfResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(
    payload: LangGraphPdfRequest,
    db: DbSession,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> LangGraphPdfResponse:
    """用 LangGraph 线性图完成「加载 → 切分 → 向量化 → 入库」。"""
    path = _resolve_pdf_path(payload.path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"PDF 文件不存在: {payload.path}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chunk_overlap 必须小于 chunk_size")

    # 把原始 PDF 上传 OSS，得到 object key（写进 source_path 供在线查询签名访问）
    try:
        object_key = await run_in_threadpool(upload_and_get_key, path)
    except Exception as exc:  # noqa: BLE001 - 上传失败统一映射为 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"原始 PDF 上传 OSS 失败: {payload.path}",
        ) from exc

    app = build_ingest_graph(db, chunk_size, chunk_overlap)
    initial_state: IngestState = {
        "path": str(path),
        "object_key": object_key,
        "documents": [],
        "chunks": [],
        "embeddings": [],
        "document_id": 0,
        "pages": 0,
        "total_chars": 0,
    }
    result = await app.ainvoke(initial_state)

    chunks: list[Document] = result["chunks"]
    embeddings: list[list[float]] = result["embeddings"]
    previews: list[dict[str, int | str]] = [
        {
            "chunk_index": index,
            "chars": len(chunk.page_content),
            "vector_dim": len(embedding),
            "preview": chunk.page_content[:60],
        }
        for index, (chunk, embedding) in enumerate(list(zip(chunks, embeddings, strict=True))[:3])
    ]

    return LangGraphPdfResponse(
        filename=path.name,
        pages=result["pages"],
        total_chars=result["total_chars"],
        chunk_count=len(chunks),
        document_id=result["document_id"],
        previews=previews,
    )
