"""千问视觉模型调用封装。

支持单张图片识别，也支持把 PDF 每页渲染成图片后逐页识别，
用于处理包含图片、表格的复杂 PDF。
"""

from base64 import b64encode
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import dashscope
from dashscope.api_entities.dashscope_response import MultiModalConversationResponse

from .config import settings

# 初始化 API Key（只需执行一次）
dashscope.api_key = settings.DASHSCOPE_API_KEY  # type: ignore[assignment]


def _call_vision(messages: list[dict[str, Any]]) -> str:
    """调用千问视觉模型，并从响应中提取文本。"""
    response = cast(
        MultiModalConversationResponse,
        dashscope.MultiModalConversation.call(
            model=settings.DASHSCOPE_VL_MODEL,
            messages=messages,
        ),
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"视觉识别失败: {response.code} - {response.message}")

    content = response.output.choices[0].message.content
    if isinstance(content, str):
        return content
    # 多模态接口返回的 content 通常形如 [{"text": "..."}]
    return "".join(item.get("text", "") for item in content if isinstance(item, dict))


def describe_image(image_path: str | Path, prompt: str = "请详细描述这张图片的内容。") -> str:
    """调用千问视觉模型识别本地图片，返回文本描述。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{Path(image_path).resolve()}"},
                {"text": prompt},
            ],
        }
    ]
    return _call_vision(messages)


def describe_image_bytes(image_bytes: bytes, prompt: str, mime: str = "image/png") -> str:
    """直接识别内存中的图片字节，返回文本描述。"""
    data_url = f"data:{mime};base64,{b64encode(image_bytes).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"image": data_url},
                {"text": prompt},
            ],
        }
    ]
    return _call_vision(messages)


def describe_pdf(
    pdf_path: str | Path,
    dpi: int = 200,
    prompt: str = "请提取这一页的全部文字，表格请用 markdown 表格表示，并描述其中的图片。",
) -> list[str]:
    """把 PDF 每页渲染成图片后，逐页交给千问视觉模型识别。

    Args:
        pdf_path: PDF 文件路径。
        dpi: 每页渲染分辨率，默认 200，越高越清晰但越慢。
        prompt: 每页识别的提示词。

    Returns:
        每一页的文本描述列表。
    """
    import fitz  # 延迟导入，仅在需要识别 PDF 时才依赖 pymupdf

    document = fitz.open(str(pdf_path))
    try:
        descriptions: list[str] = []
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi)
            descriptions.append(describe_image_bytes(pixmap.tobytes("png"), prompt))
        return descriptions
    finally:
        document.close()