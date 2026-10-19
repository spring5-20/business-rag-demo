from fastapi import APIRouter

from .v1 import router as v1_router
from .v1.langgraph_approval import router as langgraph_approval_router
from .v1.langgraph_ask import router as langgraph_ask_router
from .v1.rag_document_pdf_langgraph import router as rag_document_pdf_langgraph_router
from .v1.rag_document_pdf_online import router as rag_document_pdf_online_router
from .v1.rag_documents import router as rag_documents_router
from .v1.rag_image_documents import router as rag_image_documents_router
from .v1.rag_multipdf_documents import router as rag_multipdf_documents_router

router = APIRouter(prefix="/api")
router.include_router(v1_router)
router.include_router(rag_documents_router)
router.include_router(rag_document_pdf_online_router)
router.include_router(rag_image_documents_router)
router.include_router(rag_multipdf_documents_router)
router.include_router(rag_document_pdf_langgraph_router)
router.include_router(langgraph_ask_router)
router.include_router(langgraph_approval_router)
