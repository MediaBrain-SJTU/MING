# GitHub Actions Workflows

本目录包含项目的 CI/CD 工作流配置文件。

## 工作流说明

| 文件名 | 说明 | 触发条件 |
|--------|------|----------|
| `ci.yml` | CI 流水线 | 代码提交到非主分支、Pull Request |
| `cd.yml` | CD 流水线 | 代码合并到主分支、Tag 创建 |
| `special-scenarios.yml` | 特殊场景验证 | 修改特定目录文件 |

## 注意

**本目录仅存放 `.yml` 工作流配置文件**，其他说明文档请放在上级目录或项目根目录。
