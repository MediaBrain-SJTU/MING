# MING-7B 中文医疗大模型 - 用户指南

## 项目概述

MING-7B 是一个基于 Python 生态、PyTorch 框架和 HuggingFace 工具链构建的中文医疗大语言模型。本项目通过特征工程优化和定向微调，旨在提升专科领域（如心血管、神经内科）的问答准确率。

### 项目目标

- 专科问题准确率提升 15% 以上
- 推理速度提升 20%
- 单卡训练显存不超过 24GB
- 推理显存不超过 16GB

## 目录结构

```
MING-doubao/
├── ming/
│   ├── feature_engineering/    # 特征工程模块
│   │   ├── entity_recognizer.py    # 医疗实体识别
│   │   ├── feature_extractor.py    # 特征提取器
│   │   ├── feature_selector.py     # 特征选择器
│   │   ├── feature_pipeline.py     # 特征处理流水线
│   │   └── __init__.py
│   ├── training/               # 训练模块
│   │   ├── specialty_trainer.py    # 专科训练器
│   │   ├── convergence_optimizer.py # 收敛优化器
│   │   ├── data_pipeline.py        # 数据流水线
│   │   ├── memory_monitor.py       # 显存监控
│   │   └── __init__.py
│   ├── evaluation/             # 评估模块
│   │   ├── metrics.py              # 评估指标
│   │   ├── evaluator.py            # 评估器
│   │   ├── specialty_evaluator.py  # 专科评估器
│   │   ├── report_generator.py     # 报告生成器
│   │   └── __init__.py
│   ├── config/                 # 配置模块
│   │   ├── config_loader.py        # 配置加载器
│   │   ├── training_config.py      # 训练配置
│   │   ├── evaluation_config.py    # 评估配置
│   │   ├── feature_config.py       # 特征配置
│   │   └── __init__.py
│   ├── model/                  # 模型模块（原有结构）
│   │   ├── builder.py              # 模型构建器
│   │   └── ...
│   ├── serve/                  # 服务模块（原有结构）
│   │   ├── inference.py            # 推理接口
│   │   └── ...
│   ├── eval/                   # 评估数据
│   │   └── datasets/              # 标注数据集
│   └── main.py                 # 主入口脚本
├── configs/                    # 配置文件目录
│   ├── training_config.yaml        # 训练配置
│   ├── evaluation_config.yaml      # 评估配置
│   ├── feature_config.yaml         # 特征配置
│   └── default_config.yaml         # 默认配置
├── tests/                      # 测试目录
├── docs/                       # 文档目录
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 安装额外依赖
pip install jieba pyyaml torch transformers
```

### 2. 特征工程

```python
from ming.feature_engineering import (
    MedicalEntityRecognizer,
    FeatureExtractor,
    FeatureSelector,
    batch_feature_extraction
)

# 初始化实体识别器
recognizer = MedicalEntityRecognizer()

# 识别医疗实体
text = "患者有高血压病史，正在服用硝苯地平"
entities = recognizer.extract_entities(text)
print(f"识别实体: {entities}")

# 特征提取
extractor = FeatureExtractor()
features = extractor.extract_features(text, entities)
print(f"提取特征: {features.keys()}")

# 特征选择
selector = FeatureSelector()
selected_features = selector.select_features(features, method='mutual_info')
print(f"选择特征: {len(selected_features)} 个")
```

### 3. 模型训练

```python
from ming.training import (
    SpecialtyTrainer,
    SpecialtyTrainingArguments,
    MemoryMonitor
)
from ming.config import TrainingConfig

# 加载配置
config = TrainingConfig.from_yaml("configs/training_config.yaml")

# 初始化训练参数
args = SpecialtyTrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    learning_rate=config.learning_rate,
    max_grad_norm=config.max_grad_norm,
)

# 初始化显存监控
memory_monitor = MemoryMonitor(
    max_memory_gb=22,
    auto_adjust_batch_size=True
)

# 创建训练器
trainer = SpecialtyTrainer(
    args=args,
    memory_monitor=memory_monitor,
    target_specialties=["cardiovascular", "neurology"],
)

# 开始训练
trainer.train()
```

### 4. 模型评估

```python
from ming.evaluation import (
    ModelEvaluator,
    EvaluationConfig,
    ReportGenerator,
    SpecialtyEvaluator,
    create_default_benchmark
)

# 加载评估配置
config = EvaluationConfig.from_yaml("configs/evaluation_config.yaml")

# 创建基准测试
benchmark = create_default_benchmark(
    data_dir="ming/eval/datasets",
    baseline_em=0.65,
    target_improvement=0.15,
)

# 初始化专科评估器
evaluator = SpecialtyEvaluator(
    config=config,
    benchmark=benchmark,
)

# 运行评估
results = evaluator.evaluate(model_path="./output/model")

# 生成评估报告
generator = ReportGenerator(output_dir="./evaluation_results")
report = generator.generate_report(results, format="json")
print(f"报告生成完成: {report['report_path']}")
print(f"专科EM值提升: {results.get('improvement_rate', 0):.1%}")
```

### 5. 完整流水线

```bash
# 运行完整流水线
python ming/main.py \
    --config configs/default_config.yaml \
    --mode all \
    --output_dir ./output
```

## 配置说明

### 训练配置 (training_config.yaml)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `"./output"` | 输出目录 |
| `num_train_epochs` | `3` | 训练轮数 |
| `per_device_train_batch_size` | `8` | 单卡训练批量大小 |
| `learning_rate` | `2e-5` | 学习率 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `gradient_accumulation_steps` | `4` | 梯度累积步数 |
| `warmup_ratio` | `0.1` | 预热比例 |
| `lora_r` | `8` | LoRA 秩 |
| `lora_alpha` | `32` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |

### 评估配置 (evaluation_config.yaml)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eval_batch_size` | `8` | 评估批量大小 |
| `max_length` | `2048` | 最大序列长度 |
| `max_new_tokens` | `512` | 最大生成 token 数 |
| `target_specialties` | 5个专科 | 目标专科列表 |
| `target_improvement` | `0.15` | 目标提升率 |

### 特征配置 (feature_config.yaml)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `"cuda"` | 计算设备 |
| `batch_size` | `32` | 特征提取批量大小 |
| `max_extraction_time_ms` | `50` | 单条最大提取时间 |
| `entity_types` | 9种实体 | 医疗实体类型列表 |

## 验收标准

### 特征工程

- ✅ 实体识别 F1 值 >= 0.92
- ✅ 特征提取耗时 <= 50ms/条
- ✅ 特征覆盖率 >= 95%（覆盖95%的医疗实体类型）

### 模型训练

- ✅ 专科领域 EM 值提升 >= 15%
- ✅ 训练显存峰值 <= 22GB（单卡 A100）
- ✅ 训练收敛速度提升 >= 20%（epoch 数减少）

### 评估指标

- ✅ 评估结果复现率 = 100%
- ✅ 报告生成时间 <= 5 分钟
- ✅ 指标计算正确性通过所有测试用例

## API 接口说明

### 特征工程模块

#### MedicalEntityRecognizer

```python
class MedicalEntityRecognizer:
    def __init__(self, config: FeatureConfig = None):
        """初始化医疗实体识别器"""
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取文本中的医疗实体"""
    
    def batch_extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """批量提取医疗实体"""
    
    def get_entity_types(self) -> List[str]:
        """获取支持的实体类型"""
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估实体识别性能"""
```

#### FeatureExtractor

```python
class FeatureExtractor:
    def __init__(self, config: FeatureConfig = None):
        """初始化特征提取器"""
    
    def extract_features(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取单条样本特征"""
    
    def batch_extract_features(self, texts: List[str], entity_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """批量提取特征"""
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
```

#### FeatureSelector

```python
class FeatureSelector:
    def __init__(self, config: FeatureConfig = None):
        """初始化特征选择器"""
    
    def select_features(self, features: Dict[str, Any], method: str = "mutual_info") -> Dict[str, Any]:
        """选择重要特征"""
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
```

### 训练模块

#### SpecialtyTrainer

```python
class SpecialtyTrainer:
    def __init__(self, args: SpecialtyTrainingArguments, memory_monitor: MemoryMonitor, target_specialties: List[str]):
        """初始化专科训练器"""
    
    def train(self, resume_from_checkpoint: bool = False):
        """开始训练"""
    
    def save_model(self, output_dir: str = None):
        """保存模型"""
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
```

#### MemoryMonitor

```python
class MemoryMonitor:
    def __init__(self, max_memory_gb: float = 22.0, auto_adjust_batch_size: bool = True):
        """初始化显存监控器"""
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前显存使用情况"""
    
    def suggest_batch_size(self, current_batch_size: int) -> int:
        """根据显存使用情况建议批量大小"""
```

#### ConvergenceOptimizer

```python
class ConvergenceOptimizer:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化收敛优化器"""
    
    def apply_ema(self, model: torch.nn.Module):
        """应用指数移动平均"""
    
    def apply_swa(self, model: torch.nn.Module, swa_start: int, swa_freq: int):
        """应用随机权重平均"""
    
    def should_early_stop(self, metric_history: List[float], patience: int = 3) -> bool:
        """判断是否应该早停"""
```

### 评估模块

#### ModelEvaluator

```python
class ModelEvaluator:
    def __init__(self, config: EvaluationConfig, model, tokenizer):
        """初始化模型评估器"""
    
    def evaluate(self, eval_dataset) -> EvaluationResult:
        """运行评估"""
    
    def batch_evaluate(self, eval_datasets: Dict[str, Any]) -> Dict[str, EvaluationResult]:
        """批量评估多个数据集"""
```

#### SpecialtyEvaluator

```python
class SpecialtyEvaluator:
    def __init__(self, config: EvaluationConfig, benchmark: SpecialtyBenchmark):
        """初始化专科评估器"""
    
    def evaluate(self, model_path: str) -> SpecialtyEvaluationSummary:
        """评估专科性能"""
    
    def compare_with_baseline(self, results: SpecialtyEvaluationSummary) -> Dict[str, Any]:
        """与基线比较"""
```

#### ReportGenerator

```python
class ReportGenerator:
    def __init__(self, output_dir: str = "./evaluation_results"):
        """初始化报告生成器"""
    
    def generate_report(self, results: SpecialtyEvaluationSummary, format: str = "json") -> Dict[str, Any]:
        """生成评估报告"""
    
    def save_report(self, report: Dict[str, Any], file_path: str, format: str = "json"):
        """保存报告到文件"""
```

## 常见问题

### Q1: 如何添加新的专科数据集？

A1: 在 `configs/evaluation_config.yaml` 的 `datasets` 配置中添加新专科的数据集路径，然后在 `target_specialties` 中添加对应的专科名称。

### Q2: 训练时显存不足怎么办？

A2: 可以通过以下方式优化：
1. 减小 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 启用 `gradient_checkpointing`
4. 使用 4-bit/8-bit 量化

### Q3: 如何自定义评估指标？

A3: 在 `ming/evaluation/metrics.py` 中添加新的指标计算函数，并在 `evaluator.py` 中注册使用。

### Q4: 如何使用自己的基础模型？

A4: 修改 `configs/default_config.yaml` 中的 `base_model` 配置，确保模型与 Qwen 架构兼容。

## 技术支持

如有问题，请联系项目维护团队或提交 Issue。
