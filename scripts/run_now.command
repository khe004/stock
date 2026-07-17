#!/bin/bash
# 手动立即跑一次每日流程（拉数据 → 出信号 → 推送）。Finder 双击可用。
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python run_daily.py
echo
echo "按回车关闭窗口"
read -r
