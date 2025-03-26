#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # 根据需要调整使用的GPU

# 创建日志目录
mkdir -p logs

# 确保依赖已安装
pip install -r requirements.txt

echo "Starting RAG API server on port 50052..."

# 使用gunicorn配置文件启动服务
exec gunicorn -c gunicorn_config.py main:app 