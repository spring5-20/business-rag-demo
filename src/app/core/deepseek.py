"""DeepSeek 大模型流式对话客户端。

DeepSeek 提供与 OpenAI 兼容的 Chat Completions 接口，这里通过 httpx 直接调用，
并以异步生成器逐段返回模型生成的文本。
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .config import settings


async def stream_chat(messages: list[dict[str, str]], *, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    """流式调用 DeepSeek 对话接口，逐段产出模型回复文本。

    Args:
        messages: 对话消息列表，每个元素包含 ``role`` 与 ``content``。
        temperature: 采样温度，默认 0.7。
    """
    api_key = settings.DEEPSEEK_API_KEY
    model = settings.DEEPSEEK_MODEL
    base_url = settings.DEEPSEEK_BASE_URL
    if not api_key or not model or not base_url:
        raise RuntimeError("DeepSeek 配置不完整，请检查 DEEPSEEK_API_KEY / DEEPSEEK_MODEL / DEEPSEEK_BASE_URL")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }

    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0)) as client,
        client.stream("POST", url, headers=headers, json=payload) as response,
    ):
        if response.status_code != httpx.codes.OK:
            body = await response.aread()
            raise RuntimeError(f"DeepSeek 请求失败: {response.status_code} - {body.decode('utf-8', errors='replace')}")

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                yield content
