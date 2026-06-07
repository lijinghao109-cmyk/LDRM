#!/bin/bash
set -e

cd "$(dirname "$0")"

# 创建虚拟环境并安装依赖（如果尚未创建）
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt pyinstaller

# 打包成 macOS 独立应用
./venv/bin/pyinstaller --clean --windowed --name LDRM --onedir main.py

echo "macOS app bundle created at dist/LDRM.app"
