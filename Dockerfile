# syntax=docker/dockerfile:1.7

# --------- 构建依赖 ---------
FROM ghcr.io/astral-sh/uv:0.12.9-python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 先复制依赖清单，利用镜像层缓存
COPY pyproject.toml uv.lock ./
COPY .python-version ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# --------- 运行镜像 ---------
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./
COPY .python-version ./
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# -------- 使用 gunicorn 时替换下面的命令 --------
# CMD ["gunicorn", "src.app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
