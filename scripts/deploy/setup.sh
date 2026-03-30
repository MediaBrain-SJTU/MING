#!/bin/bash
# 部署脚本 - 设置医疗大模型优化部署环境

set -e

echo "=========================================="
echo "MING医疗大模型优化部署脚本"
echo "=========================================="

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 安装额外的优化依赖
echo "安装优化依赖..."
pip install bitsandbytes==0.41.0
pip install accelerate==0.27.2
pip install peft==0.8.2
pip install transformers==4.37.0

# 创建必要的目录
echo "创建项目目录..."
mkdir -p output
mkdir -p eval_results
mkdir -p logs
mkdir -p cache

# 设置环境变量
echo "设置环境变量..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="$(pwd)/cache"
export TRANSFORMERS_CACHE="$(pwd)/cache"

# 检查GPU可用性
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "警告: 未检测到NVIDIA GPU"
fi

echo "=========================================="
echo "部署完成!"
echo "=========================================="
echo ""
echo "使用说明:"
echo "1. 激活环境: source venv/bin/activate"
echo "2. 运行测试: python -m pytest tests/ -v"
echo "3. 启动训练: python scripts/train.py --config configs/training_config.yaml"
echo "4. 运行评估: python scripts/evaluate.py --config configs/eval_config.yaml"
echo ""
