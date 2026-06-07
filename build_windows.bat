@echo off
REM 请在 Windows 环境中运行此脚本。
REM 需要先创建虚拟环境并安装依赖：
REM python -m venv venv
REM venv\Scripts\activate
REM pip install -r requirements.txt pyinstaller

python -m pyinstaller --clean --windowed --onefile --name LDRM main.py

echo Windows 应用已生成，请查看 dist\LDRM.exe
pause
