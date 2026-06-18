#!/bin/bash
# qkquant 服务器部署脚本
# 适用: Ubuntu 22.04 / CentOS 7+ / Debian 11+
# 硬件: 1核 1G+ 即可

set -e

echo "=== qkquant 服务器部署 ==="
echo ""

# 1. 基础环境
echo "[1/5] 安装依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv git curl

# 2. 创建项目
echo "[2/5] 拉取项目..."
cd /opt
sudo mkdir -p qkquant
sudo chown $USER:$USER qkquant
# 从本地上传项目文件（scp 整个目录）
# scp -r d:/lsl/qkquant/* user@server:/opt/qkquant/

echo "  请手动 scp 项目文件到 /opt/qkquant/"
echo "  scp -r d:/lsl/qkquant/* user@server:/opt/qkquant/"
echo ""

# 3. Python 虚拟环境
echo "[3/5] 配置 Python 环境..."
cd /opt/qkquant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 2>/dev/null || pip install \
    pandas numpy duckdb akshare baostock backtrader \
    matplotlib pydantic pyyaml loguru typer rich \
    tenacity tqdm streamlit plotly requests

# 4. 初始化数据
echo "[4/5] 首次拉数据（约需 30 分钟）..."
source .venv/bin/activate
python -m qkquant.cli update-data --universe hs300 --full

# 5. 定时任务
echo "[5/5] 配置 crontab..."
cat << 'CRON' | crontab -
# qkquant 每日扫描 — 交易日 18:30
30 18 * * 1-5 cd /opt/qkquant && .venv/bin/python -m qkquant.cli scan --raw --push --auto-position --ai >> logs/cron.log 2>&1
CRON

echo ""
echo "=== 部署完成 ==="
echo "手动步骤:"
echo "  1. scp 项目文件到服务器: scp -r ./* user@server:/opt/qkquant/"
echo "  2. 配置 config/notify.yaml 推送通道"
echo "  3. 配置 config/ai.yaml DeepSeek API Key"
echo "  4. 首次人工跑一次确认:"
echo "     cd /opt/qkquant && source .venv/bin/activate"
echo "     python -m qkquant.cli scan --raw"
echo "  5. 启动面板: streamlit run scripts/dashboard.py --server.port 8501"
