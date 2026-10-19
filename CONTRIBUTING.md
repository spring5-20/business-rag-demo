# 贡献指南

感谢你对 **FastAPI Template** 的关注！无论是报告 Bug、改进文档，还是提交代码，我们都非常欢迎。

## 准备开发环境

### 克隆仓库

1. **Fork 仓库**：在目标平台（如 Gitee / GitHub）上 Fork 本项目。
2. **创建分支**：基于最新主分支创建特性分支：

   ```sh
   git checkout -b feature/fooBar
   ```

3. **安装依赖**：本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖，请先安装 uv（安装方式见 `README.md` 的「安装 uv」章节）。

   ```sh
   uv sync
   ```

## 开发与验证

### 启动服务

```sh
uv run python src/run.py
```

### 运行测试

```sh
uv run python -m pytest
```

### 代码风格与静态检查

```sh
uv run ruff check --fix
uv run ruff format
uv run mypy src
```

### 提交前检查

建议安装 pre-commit，在提交时自动执行检查：

```sh
uv run pre-commit install
```

## 编码规范

- 遵循 PEP 8，行宽 120，缩进 4 空格。
- 文件、函数、变量使用 `snake_case`，类使用 `PascalCase`，数据库列使用 `snake_case`。
- 路由统一以 `/api/v1/` 为前缀。
- 测试文件命名为 `test_<entity>.py`，新增功能请同步补充测试。
- 公共函数与类补充中文 docstring，并尽量标注类型。

## 提交 Pull Request

1. 将改动推送到你的 Fork 分支：

   ```sh
   git push origin feature/fooBar
   ```

2. 打开 Pull Request，说明改动目的，关联相关 issue。
3. 必要时更新 `README.md`。
4. 确保测试与静态检查全部通过。

## 其他约定

- 保持与上游仓库同步，避免合并冲突。
- 遵循仓库根目录的 `AGENTS.md` 与 `CODE_OF_CONDUCT.md`。

感谢你的贡献！🚀
