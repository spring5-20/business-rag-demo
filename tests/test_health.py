import pytest
from httpx import ASGITransport, AsyncClient

from src.app.main import app


@pytest.mark.anyio
async def test_health_check() -> None:
    """健康检查接口应返回 200 与 ok 状态。

    使用 ``httpx.AsyncClient`` 与 ``ASGITransport`` 调用应用，
    不执行应用生命周期，
    从而避免在测试环境中依赖 PostgreSQL 与 Redis 服务。
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
