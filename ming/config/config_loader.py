"""
配置加载器模块

本模块提供YAML配置文件的加载、验证、合并等功能
"""

import os
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证错误"""

    pass


def load_config(
    config_path: str,
    allow_missing: bool = False,
) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径
        allow_missing: 是否允许文件不存在（不存在时返回空字典

    Returns:
        Dict[str, Any]: 配置字典

    Raises:
        FileNotFoundError: 文件不存在且allow_missing=False时
        yaml.YAMLError: YAML解析错误
    """
    path = Path(config_path)

    if not path.exists():
        if allow_missing:
            logger.warning(f"配置文件不存在，返回空配置: {config_path}")
            return {}
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config or {}
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误: {e}")
        raise


def save_config(
    config: Dict[str, Any], config_path: str, overwrite: bool = True) -> None:
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        config_path: 输出路径
        overwrite: 是否覆盖已存在的文件
    """
    path = Path(config_path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"配置文件已存在且不允许覆盖: {config_path}")

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )

    logger.info(f"配置文件已保存: {config_path}")


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
    recursive: bool = True,
) -> Dict[str, Any]:
    """
    合并两个配置字典

    Args:
        base_config: 基础配置
        override_config: 覆盖配置（优先级更高）
        recursive: 是否递归合并

    Returns:
        Dict[str, Any]: 合并后的配置
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if (
            recursive
            and key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value, recursive=True)
        else:
            result[key] = value

    return result


def validate_config(
    config: Dict[str, Any],
    schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置是否符合schema

    Args:
        config: 待验证的配置
        schema: 验证schema

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误列表]
    """
    errors = []
    _validate_object(config, schema, "", errors)
    return len(errors) == 0, errors


def _validate_object(
    obj: Any,
    schema: Dict[str, Any],
    path: str,
    errors: List[str],
) -> None:
    """递归验证配置对象"""
    if not isinstance(obj, dict):
        errors.append(f"{path}: 应为字典类型")
        return

    for key, value in schema.items():
        if key not in obj:
            schema_val = schema[key]
            if isinstance(schema_val, dict) and schema_val.get("required", False):
                errors.append(f"{path}.{key}: 缺少必填字段")
            continue

        obj_val = obj[key]
        schema_val = schema[key]

        if isinstance(schema_val, dict):
            if "type" in schema_val:
                expected_type = schema_val["type"]
                if not _check_type(obj_val, expected_type):
                    errors.append(
                        f"{path}.{key}: 类型应为 {expected_type}，实际为 {type(obj_val).__name__}"
                    )
                    continue

            if "enum" in schema_val and obj_val not in schema_val["enum"]:
                errors.append(
                    f"{path}.{key}: 值应为 {schema_val['enum']} 之一，实际为 {obj_val}"
                )

            if isinstance(obj_val, dict) and "schema" in schema_val:
                _validate_object(obj_val, schema_val["schema"], f"{path}.{key}", errors)
        elif isinstance(schema_val, dict):
            _validate_object(obj_val, schema_val, f"{path}.{key}", errors)


def _check_type(value: Any, type_name: str) -> bool:
    """检查值的类型是否符合预期"""
    type_map = {
        "str": str,
        "int": int,
        "float": (int, float),
        "bool": bool,
        "list": list,
        "dict": dict,
        "number": (int, float),
    }

    expected_types = type_map.get(type_name)
    if expected_types is None:
        return True

    return isinstance(value, expected_types)


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置

    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        "model": {
            "name": "MING-7B",
            "path": "",
            "prompt_type": "qwen",
        },
        "training": {
            "output_dir": "./output",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_memory_usage_gb": 22.0,
            "fp16": True,
            "gradient_checkpointing": True,
        },
        "evaluation": {
            "output_dir": "./evaluation_results",
            "eval_batch_size": 8,
            "max_length": 2048,
            "max_new_tokens": 512,
            "compute_entity_metrics": True,
            "save_outputs": True,
            "seed": 42,
        },
        "feature_engineering": {
            "max_entity_types": [
                "疾病",
                "症状",
                "药物",
                "检查",
                "治疗",
                "身体部位",
                "医疗器械",
                "细菌",
                "病毒",
                "专科",
            ],
            "batch_size": 32,
            "device": "cuda",
        },
        "specialties": {
            "target_specialties": [
                "cardiovascular",
                "neurology",
                "respiratory",
                "gastroenterology",
                "endocrinology",
            ],
            "weights": {
                "cardiovascular": 1.0,
                "neurology": 1.0,
                "respiratory": 1.0,
                "gastroenterology": 1.0,
                "endocrinology": 1.0,
            },
        },
    }


def main():
    """测试函数"""
    print("配置加载器模块测试")

    # 获取默认配置
    default_config = get_default_config()
    print("默认配置结构:")
    for section in default_config.keys():
        print(f"  - {section}")

    # 保存默认配置
    test_path = "./test_default_config.yaml"
    try:
        save_config(default_config, test_path)
        print(f"默认配置已保存到: {test_path}")

        # 重新加载配置
        loaded_config = load_config(test_path)
        print(f"配置加载成功，包含 {len(loaded_config)} 个配置节")

        # 验证配置
        schema = {
            "model": {
                "required": True,
                "schema": {
                    "name": {"type": "str", "required": True},
                    "path": {"type": "str"},
                },
            },
            "training": {
                "required": True,
                "schema": {
                    "output_dir": {"type": "str"},
                    "num_train_epochs": {"type": "int"},
                },
            },
        }

        is_valid, errors = validate_config(loaded_config, schema)
        if is_valid:
            print("配置验证通过")
        else:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
    finally:
        # 清理测试文件
        import os
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"测试文件已清理: {test_path}")


if __name__ == "__main__":
    main()
