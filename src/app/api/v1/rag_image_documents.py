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
from ...core.vision import describe_image

router = APIRouter(prefix="/multi", tags=["多模态图片入库"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]

# 中文友好的分隔符优先级
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class MultimodalIngestRequest(BaseModel):
    """多模态入库请求。"""

    path: str = Field(..., min_length=1, description="图片相对 DOCUMENT_DIR 的路径")
    prompt: str | None = Field(None, description="视觉识别提示词，默认使用通用描述")


class MultimodalIngestResponse(BaseModel):
    """多模态入库结果。"""

    filename: str = Field(..., description="图片文件名")
    description: str = Field(..., description="千问视觉模型识别出的文本描述")
    chars: int = Field(..., description="描述文本字符数")
    chunk_count: int = Field(..., description="切分并入库的文本块数量")
    document_id: int = Field(..., description="rag_documents 表生成的文档 id")
    previews: list[dict[str, int | str]] = Field(default_factory=list, description="前三个文本块预览")


def _resolve_image_path(path: str) -> Path:
    """把相对路径解析到 DOCUMENT_DIR 内，防止目录穿越。"""
    root = settings.DOCUMENT_DIR.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="路径必须在 DOCUMENT_DIR 目录内",
        )
    return candidate


@router.post("/image", response_model=MultimodalIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_image(
    payload: MultimodalIngestRequest,
    db: DbSession,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> MultimodalIngestResponse:
    """传入一张图片，用 LCEL 完成「视觉识别 → 切分 → 向量化 → 入库」。"""
    path = _resolve_image_path(payload.path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"图片不存在: {payload.path}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chunk_overlap 必须小于 chunk_size")

    prompt = payload.prompt or "请详细描述这张图片的内容。"

    async def recognize(image_path: Path) -> dict[str, Any]:
        """步骤 1：千问视觉模型识别图片，生成文本描述。"""
        text: str = await run_in_threadpool(describe_image, image_path, prompt)
        if not text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未能从图片中识别出内容")
        return {"description": text}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )

    def split_description(result: dict[str, Any]) -> dict[str, Any]:
        """步骤 2：用 LangChain 切分器把描述切块。"""
        result["chunks"] = splitter.split_text(result["description"])
        return result

    pairs = []
    async def vectorize(result: dict[str, Any]) -> dict[str, Any]:
        """步骤 3：向量化，返回 content 与 embedding 的配对列表。"""
        
        for chunk in result["chunks"]:
            embedding = await run_in_threadpool(get_text_embedding, chunk, EMBEDDING_DIMENSION)
            pairs.append({"content": chunk, "embedding": embedding})
        result["pairs"] = pairs
        return result

    # LCEL 链：识别 → 切分 → 向量化
    chain: Any = (
        RunnableLambda(typing_cast(Any, recognize))
        | RunnableLambda(typing_cast(Any, split_description))
        | RunnableLambda(typing_cast(Any, vectorize))
    )
    result = await chain.ainvoke(path)

    # 步骤 4（前置）：把原始图片上传 OSS，得到 object key
    try:
        object_key = await run_in_threadpool(upload_and_get_key, path)
    except Exception as exc:  # noqa: BLE001 - 上传失败统一映射为 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"原始图片上传 OSS 失败: {payload.path}",
        ) from exc

    # 步骤 4：入库（副作用，放在链外；source_path 存 OSS object key）
    document = RagDocument(title=path.stem or path.name, source_type="image", source_path=object_key)
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

    return MultimodalIngestResponse(
        filename=path.name,
        description=description,
        chars=len(description),
        chunk_count=len(pairs),
        document_id=document.id,
        previews=previews,
    )