"""复杂 PDF 视觉识别入库接口。

把 PDF 每页渲染成图片，交给千问视觉模型逐页识别（提取文字、表格转 markdown、
描述图片），再切分、向量化后入库。适用于含图片、表格的复杂 PDF。
"""

from pathlib import Path
from typing import Annotated, Any
from typing import cast as typing_cast

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db.database import async_get_db
from ...core.db.models import RagChunk, RagDocument
from ...core.embeddings import EMBEDDING_DIMENSION, get_text_embedding
from ...core.oss import upload_and_get_key
from ...core.vision import describe_pdf

router = APIRouter(prefix="/multi", tags=["复杂pdf入库"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]

# 中文友好的分隔符优先级
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class PdfVisionIngestRequest(BaseModel):
    """复杂 PDF 视觉识别入库请求。"""

    path: str = Field(..., min_length=1, description="PDF 相对 DOCUMENT_DIR 的路径")
    prompt: str | None = Field(None, description="每页视觉识别提示词，默认提取文字、表格并描述图片")
    dpi: int = Field(200, ge=72, le=600, description="每页渲染分辨率，越高越清晰但越慢")


class PdfVisionIngestResponse(BaseModel):
    """复杂 PDF 视觉识别入库结果。"""

    filename: str = Field(..., description="PDF 文件名")
    pages: int = Field(..., description="PDF 页数")
    chars: int = Field(..., description="识别出的总字符数")
    chunk_count: int = Field(..., description="切分并入库的文本块数量")
    document_id: int = Field(..., description="rag_documents 表生成的文档 id")
    previews: list[dict[str, int | str]] = Field(default_factory=list, description="前三个文本块预览")


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


@router.post("/pdf-vision", response_model=PdfVisionIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf_vision(
    payload: PdfVisionIngestRequest,
    db: DbSession,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> PdfVisionIngestResponse:
    """传入一个复杂 PDF，用 LCEL 完成「逐页视觉识别 → 切分 → 向量化 → 入库」。"""
    path = _resolve_pdf_path(payload.path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"PDF 文件不存在: {payload.path}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chunk_overlap 必须小于 chunk_size")

    prompt = payload.prompt or "请提取这一页的全部文字，表格请用 markdown 表格表示，并描述其中的图片。"

    async def recognize(pdf_path: Path) -> dict[str, Any]:
        """步骤 1：PDF 每页渲染成图片，逐页交给 qwen-vl 识别。"""
        page_descriptions = await run_in_threadpool(describe_pdf, pdf_path, payload.dpi, prompt)
        if not page_descriptions:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未能从 PDF 中识别出内容")
        description = "\n\n".join(
            f"【第 {page_number} 页】\n{page_text}"
            for page_number, page_text in enumerate(page_descriptions, start=1)
        )
        return {"pages": len(page_descriptions), "description": description}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )

    def split_step(result: dict[str, Any]) -> dict[str, Any]:
        """步骤 2：用 LangChain 切分器把识别结果切块。"""
        result["chunks"] = splitter.split_text(result["description"])
        return result

    async def vectorize_step(result: dict[str, Any]) -> dict[str, Any]:
        """步骤 3：向量化，返回 content 与 embedding 的配对列表。"""
        pairs: list[dict[str, Any]] = []
        for chunk in result["chunks"]:
            embedding = await run_in_threadpool(get_text_embedding, chunk, EMBEDDING_DIMENSION)
            pairs.append({"content": chunk, "embedding": embedding})
        result["pairs"] = pairs
        return result

    # LCEL 链：逐页识别 → 切分 → 向量化
    chain: Any = (
        RunnableLambda(typing_cast(Any, recognize))
        | RunnableLambda(typing_cast(Any, split_step))
        | RunnableLambda(typing_cast(Any, vectorize_step))
    )
    result = await chain.ainvoke(path)

    pairs: list[dict[str, Any]] = result["pairs"]

    # 步骤 4（前置）：把原始 PDF 上传 OSS，得到 object key
    try:
        object_key = await run_in_threadpool(upload_and_get_key, path)
    except Exception as exc:  # noqa: BLE001 - 上传失败统一映射为 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"原始 PDF 上传 OSS 失败: {payload.path}",
        ) from exc

    # 步骤 4：入库（副作用，放在链外；source_path 存 OSS object key）
    document = RagDocument(title=path.stem or path.name, source_type="pdf_vision", source_path=object_key)
    db.add(document)
    await db.flush()
    for index, pair in enumerate(pairs):
        db.add(
            RagChunk(
                document_id=document.id,
                chunk_index=index,
                content=pair["content"],
                embedding=pair["embedding"],
                token_count=len(pair["content"]),
            )
        )
    await db.commit()

    pages: int = result["pages"]
    description: str = result["description"]
    previews: list[dict[str, int | str]] = [
        {
            "chunk_index": index,
            "chars": len(pair["content"]),
            "vector_dim": len(pair["embedding"]),
            "preview": pair["content"][:60],
        }
        for index, pair in enumerate(pairs[:3])
    ]

    return PdfVisionIngestResponse(
        filename=path.name,
        pages=pages,
        chars=len(description),
        chunk_count=len(pairs),
        document_id=document.id,
        previews=previews,
    )
