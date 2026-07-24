#!/bin/bash
# macOS 一键部署：创建虚拟环境、安装依赖、注册 launchd 每日定时任务。
# 用法: bash scripts/setup_mac.sh [HH:MM]   默认每天 15:00（盘后）运行
# 注意：时间是本机本地时间，需确保在美股收盘后当日行情已定稿（如在美东时区，
#       15:00 是盘中、当日K线未定稿，应改到收盘后如 16:30 之后）。
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
RUN_AT="${1:-15:00}"
HOUR="${RUN_AT%%:*}"
MINUTE="${RUN_AT##*:}"
LABEL="com.quant.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "==> 项目目录: $PROJECT_DIR"

if [ ! -d .venv ]; then
    echo "==> 创建虚拟环境 .venv"
    python3 -m venv .venv
fi

echo "==> 安装依赖"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "==> 已生成 .env，如需 Telegram 推送请编辑它填入 token"
fi

mkdir -p logs "$HOME/Library/LaunchAgents"

echo "==> 写入 launchd 任务: 每天 $HOUR:$MINUTE 运行（比 cron 好在：睡眠中错过的任务会在唤醒后补跑）"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/.venv/bin/python</string>
        <string>$PROJECT_DIR/run_daily.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>$HOUR</integer>
        <key>Minute</key>
        <integer>$MINUTE</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchd.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

# 注意：macOS 自带 bash 3.2 会把紧跟在变量后的全角字符吞进变量名，
# 变量一律用 ${VAR} 花括号形式，且不用全角括号包变量
echo
echo "✅ 部署完成"
echo "   定时任务:   每天 ${HOUR}:${MINUTE} 自动运行 run_daily.py (${LABEL})"
echo "   立即试跑:   .venv/bin/python run_daily.py"
echo "   查看日志:   tail -f logs/launchd.log"
echo "   复盘面板:   .venv/bin/streamlit run quant/web/app.py"
echo "   取消任务:   launchctl unload ${PLIST} && rm ${PLIST}"
echo
echo "提示: 如需 Telegram 推送，编辑 .env 填入 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID"
