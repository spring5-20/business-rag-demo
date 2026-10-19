"""
编写pdf文档加载器
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def test_load_pdf():
    pdfs = PyPDFLoader("D:/rag知识库/测试文档.pdf").load()
    print(pdfs[0].page_content)
    print(pdfs[0].metadata)
