"""阿里云 OSS 对象存储服务封装。"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import oss2
from oss2.exceptions import OssError

from .config import settings

logger = logging.getLogger(__name__)


class OSSConfigError(RuntimeError):
    """OSS 配置不完整时抛出的异常。"""


class OSSService:
    """封装阿里云 OSS 的上传、删除、签名 URL 与列举操作。"""

    def __init__(self) -> None:
        access_key_id = settings.OSS_ACCESS_KEY_ID
        access_key_secret = settings.OSS_ACCESS_KEY_SECRET
        bucket_name = settings.OSS_BUCKET_NAME
        endpoint = settings.OSS_ENDPOINT
        if not (access_key_id and access_key_secret and bucket_name and endpoint):
            raise OSSConfigError(
                "OSS 配置不完整，请检查 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / OSS_BUCKET_NAME / OSS_ENDPOINT"
            )
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def upload_file(self, file_content: bytes, object_key: str, headers: dict[str, Any] | None = None) -> str:
        """上传文件，返回 1 小时有效的签名 URL。"""
        try:
            self.bucket.put_object(object_key, file_content, headers=headers)
        except OssError as exc:
            logger.exception("上传 OSS 失败: %s", object_key)
            raise RuntimeError(f"上传 OSS 失败: {object_key}") from exc
        return self.get_file_url(object_key)

    def upload_local_file(self, local_path: str | Path, object_key: str) -> str:
        """把本地文件上传到 OSS，返回签名 URL。"""
        with open(local_path, "rb") as file:
            return self.upload_file(file.read(), object_key)

    def delete_file(self, object_key: str) -> bool:
        """删除 OSS 对象，成功返回 True。"""
        try:
            self.bucket.delete_object(object_key)
        except OssError as exc:
            logger.exception("删除 OSS 失败: %s", object_key)
            raise RuntimeError(f"删除 OSS 失败: {object_key}") from exc
        return True

    def get_file_url(self, object_key: str, expires: int = 3600) -> str:
        """生成指定有效期的 GET 签名 URL（统一返回 https，供前端直接访问）。"""
        url = cast(str, self.bucket.sign_url("GET", object_key, expires))
        if url.startswith("http://"):
            # oss2 依 endpoint 未带协议时默认 http；OSS 同时支持 https，签名不依赖 scheme，
            # 直接替换可避免前端 HTTPS 页面出现混合内容告警。
            url = "https://" + url[len("http://") :]
        return url

    def list_files(self, prefix: str = "", max_keys: int = 100) -> list[dict[str, Any]]:
        """列举 OSS 对象，返回 key、size、last_modified。"""
        result = self.bucket.list_objects(prefix=prefix, max_keys=max_keys)
        return [{"key": obj.key, "size": obj.size, "last_modified": obj.last_modified} for obj in result.object_list]


@lru_cache
def get_oss_service() -> OSSService:
    """获取 OSS 服务单例。"""
    return OSSService()


def build_object_key(local_path: str | Path) -> str:
    """把本地文件路径转成相对 DOCUMENT_DIR 的 OSS object key。"""
    relative = Path(local_path).resolve().relative_to(settings.DOCUMENT_DIR.resolve())
    return str(relative).replace("\\", "/")


def upload_to_oss(local_path: str | Path, object_key: str) -> str:
    """上传本地文件到 OSS，返回签名 URL。"""
    return get_oss_service().upload_local_file(local_path, object_key)


def upload_and_get_key(local_path: str | Path) -> str:
    """上传原始文件到 OSS 并返回 object key（供入库写 source_path）。

    内部先按相对 DOCUMENT_DIR 的相对路径生成 object key，再上传；
    调用方只需关心返回的 key，无需自己拼签名 URL。
    """
    object_key = build_object_key(local_path)
    get_oss_service().upload_local_file(local_path, object_key)
    return object_key


def safe_get_file_url(object_key: str | None, expires: int = 3600) -> str | None:
    """尽力生成指定有效期的 GET 签名 URL；OSS 未配置或出错时返回 None。

    供在线查询接口对命中的 object key 生成临时可访问 URL——失败时优雅降级，
    返回 None，不影响问答主流程。
    """
    if not object_key:
        return None
    try:
        return get_oss_service().get_file_url(object_key, expires)
    except Exception:  # noqa: BLE001 - OSS 不可用时优雅降级
        logger.exception("生成 OSS 签名 URL 失败: %s", object_key)
        return None


def looks_like_object_key(value: str | None) -> bool:
    """粗略判断存量 source_path 是否为 OSS object key 形态。

    用于排除旧数据的本地绝对路径（Windows 反斜杠 / POSIX 前导斜杠 / 盘符），
    避免对本地路径生成指向不存在对象的假签名 URL。
    """
    if not value:
        return False
    # 排除旧数据的本地绝对路径：Windows 反斜杠 / POSIX 前导斜杠 / 盘符开头
    return not ("\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value))