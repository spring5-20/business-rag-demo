import warnings
from collections.abc import Callable, Generator
from typing import Any

# starlette 的 TestClient 在上游切换到 httpx2 之前会触发可忽略的弃用提示，
# 因此在导入前先抑制该警告，避免污染测试输出。
warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message="The anyio.abc.BlockingPortal alias is deprecated.*",
    category=Warning,
)

import pytest  # noqa: E402
from faker import Faker  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.orm.session import Session  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from src.app.core.config import settings  # noqa: E402
from src.app.main import app  # noqa: E402

DATABASE_URI = settings.POSTGRES_URI
DATABASE_PREFIX = settings.POSTGRES_SYNC_PREFIX

sync_engine = create_engine(DATABASE_PREFIX + DATABASE_URI)
local_session = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


fake = Faker()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, Any, None]:
    with TestClient(app) as _client:
        yield _client
    app.dependency_overrides = {}
    sync_engine.dispose()


@pytest.fixture
def db() -> Generator[Session, Any, None]:
    session = local_session()
    yield session
    session.close()


def override_dependency(dependency: Callable[..., Any], mocked_response: Any) -> None:
    app.dependency_overrides[dependency] = lambda: mocked_response
