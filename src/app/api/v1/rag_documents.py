from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db.database import async_get_db
from ...core.db.models import RagChunk, RagDocument
from ...core.embeddings import get_text_embedding
from ...core.oss import upload_and_get_key

router = APIRouter(prefix="/rag", tags=["RAG最小化演示"])

DbSession = Annotated[AsyncSession, Depends(async_get_db)]


class PdfIngestRequest(BaseModel):  # noqa: F821
    """PDF 入库请求。"""

    path: str = Field(..., min_length=1, description="PDF 相对 DOCUMENT_DIR 的路径")  # noqa: F821


class PdfIngestResponse(BaseModel):
    """入库结果，方便课堂观察每一步的输出。"""

    filename: str = Field(..., description="PDF 文件名")
    pages: int = Field(..., description="PDF 页数")
    total_chars: int = Field(..., description="提取出的总字符数")
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

"""
编写pdf文档加载器
"""

def load_pdf(path:Path) -> list[Document]:
    return PyPDFLoader(str(path)).load()

"""
编写pdf文本分割器，采用递归语义文本分割器
"""

# 中文友好的分隔符优先级：先按段落、换行，再按中文标点，最后按空格与单个字符兜底
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """步骤 2：用 LangChain 的 RecursiveCharacterTextSplitter 切分文本。"""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    splitter = RecursiveCharacterTextSplitter(  # noqa: F821
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    return splitter.split_documents(documents)


#编写文本向量化函数
async def embed_chunk(text:str) -> list[float]:
    return await run_in_threadpool(get_text_embedding,text,1536)


"""保存向量数据入库"""

async def save_to_db(
        db: AsyncSession,
        title:str,
        source_path:str,
        chunks:list[Document],
        embeddings:list[list[float]]
) -> tuple[int,int]:

    # 先保存 rag_document
    rag_document = RagDocument(title=title, source_path=source_path, source_type="pdf")
    db.add(rag_document)
    await db.flush()
    # 再保存 rag_document 表对应的文档 chunk 表
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        db.add(
            RagChunk(
                document_id=rag_document.id,
                chunk_index=index,
                content=chunk.page_content,
                embedding=embedding,
                token_count=len(chunk.page_content),
            )
        )
    await db.commit()
    return rag_document.id, len(chunks)



@router.post("/pdf", response_model=PdfIngestResponse, status_code=status.HTTP_201_CREATED)  # noqa: F821
async def ingest_pdf(
    payload: PdfIngestRequest,
    db: DbSession,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> PdfIngestResponse | None:
    """传入一个 PDF 路径，一次完成「加载 → 切分 → 向量化 → 入库」四步。"""

    path = _resolve_pdf_path(payload.path)  # noqa: F821

    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"PDF 文件不存在: {payload.path}")  # noqa: F821

    # 步骤 1：LangChain 加载 PDF
    documents = load_pdf(path)
    if not documents or not "".join(document.page_content for document in documents).strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未能从 PDF 中提取到文本")

    # 步骤 2：LangChain 切分
    try:
        chunks = split_documents(documents, chunk_size, chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # 步骤 3：向量化
    embeddings = [await embed_chunk(chunk.page_content) for chunk in chunks]

    # 步骤 4：把原始 PDF 上传 OSS，得到 object key（存 source_path）
    try:
        object_key = await run_in_threadpool(upload_and_get_key, path)
    except Exception as exc:  # noqa: BLE001 - 上传失败统一映射为 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"原始文件上传 OSS 失败: {payload.path}",
        ) from exc

    # 步骤 5：入库（source_path 存 OSS object key）
    document_id, chunk_count = await save_to_db(db, path.stem or path.name, object_key, chunks, embeddings)


    previews: list[dict[str, int | str]] = [
        {
            "chunk_index": index,
            "chars": len(chunk.page_content),
            "vector_dim": len(embedding),
            "preview": chunk.page_content[:60],
        }
        for index, (chunk, embedding) in enumerate(list(zip(chunks, embeddings, strict=True))[:3])
    ]

    return PdfIngestResponse(
        filename=path.name,
        pages=len(documents),
        total_chars=sum(len(document.page_content) for document in documents),
        chunk_count=chunk_count,
        document_id=document_id,
        previews=previews,
    )