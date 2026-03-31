# CI/CD 配置说明

## 工作流文件说明

### 1. ci.yml - CI 流水线
- **触发条件**: 代码提交到非 main/master 分支，或创建 Pull Request
- **功能**:
  - 代码质量检查 (Ruff + Black)
  - 单元测试 (多 Python 版本: 3.8, 3.9, 3.10)
  - 集成测试
  - 构建验证

### 2. cd.yml - CD 流水线
- **触发条件**: 代码合并到 main/master 分支，或创建 Tag (v*)
- **功能**:
  - 版本号验证 (语义化版本检查)
  - 包发布到 PyPI
  - GitHub Release 创建
  - 部署通知

### 3. special-scenarios.yml - 特殊场景流水线
- **触发条件**: 修改特定目录下的文件
- **功能**:
  - **训练任务触发** (ming/train/ 或 scripts/*.json):
    - DeepSpeed 配置文件格式验证
    - 训练脚本 dry-run
    - GPU/内存资源配置检查
  - **评估任务触发** (ming/eval/):
    - 评估数据集格式验证
    - 评估指标计算逻辑测试
  - **服务部署触发** (ming/serve/):
    - FastAPI 服务启动测试
    - Gradio 界面可用性测试
    - API 接口测试

## 需要配置的 Secrets

在 GitHub 项目的 Settings > Secrets and variables > Actions 中添加以下 Secrets:

| Secret 名称 | 说明 | 是否必填 |
|-----------|------|---------|
| `PYPI_TOKEN` | PyPI 访问令牌，用于发布包 | CD 流水线必需 |
| `CODECOV_TOKEN` | Codecov 令牌，用于测试覆盖率报告 | 可选 |
| `GITHUB_TOKEN` | GitHub 令牌，自动创建，无需手动添加 | 必需 |

## 本地开发命令

```bash
# 安装开发依赖
pip install -e .[dev]

# 代码质量检查
ruff check ming/
black ming/ --check

# 运行单元测试
pytest tests/unit -v

# 运行集成测试
pytest tests/integration -v

# 构建包
python -m build

# 检查包
twine check dist/*
```

## 版本发布流程

1. 更新 `pyproject.toml` 中的版本号
2. 提交代码并创建 Tag:
```bash
git tag v1.1.4
git push origin v1.1.4
```
3. CD 流水线会自动触发，完成版本验证、包发布和 Release 创建
