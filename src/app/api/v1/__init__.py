from fastapi import APIRouter

router = APIRouter(prefix="/v1")


@router.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """存活探针，用于快速验证服务是否正常启动。"""
    return {"status": "ok"}


# 在这里按业务模块注册 v1 路由，例如：
# from . import health
# router.include_router(health.router)
