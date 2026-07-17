#!/bin/bash
# 启动复盘面板。Finder 里双击即可（.command 会自动打开终端运行）。
# 用脚本自身位置定位仓库根目录，仓库移动/重克隆后无需修改。
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/streamlit run quant/web/app.py
