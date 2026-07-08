"""Telegram Bot 推送。未配置 token 时降级为打印到终端。"""

import logging
import os
import time

import requests

log = logging.getLogger(__name__)

MAX_RETRIES = 3


def send_message(text: str) -> bool:
    """发送消息，返回“是否已送达”。未配置 token 时打印到终端并视为已送达（避免每次
    运行重复输出同一批信号）；仅网络发送失败返回 False，信号会留到下次运行重试。"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("未配置 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID，消息仅打印到终端")
        print(text)
        return True

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            last_err = e
            wait = 2**attempt
            log.warning("Telegram 第 %d 次发送失败: %s，%ds 后重试", attempt, e, wait)
            time.sleep(wait)
    log.error("Telegram 发送失败: %s，消息打印到终端兜底", last_err)
    print(text)
    return False
