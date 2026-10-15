"""微信推送（Server酱，sct.ftqq.com）：消息进个人微信。未配置 SendKey 时不启用该渠道。

Server酱 的推送分两部分：title（微信消息列表里看到的标题，上限 32 字）和 desp（点开后的
正文，Markdown）。日报正文是纯文本，Markdown 会把单个换行合并成一行，所以发送前把每个
换行改成 Markdown 的硬换行。
"""

import logging
import os
import re
import time

import requests

log = logging.getLogger(__name__)

MAX_RETRIES = 3
TITLE_LIMIT = 32


def sendkey() -> str | None:
    return (os.getenv("SERVERCHAN_SENDKEY") or "").strip() or None


def api_url(key: str) -> str:
    """Server酱³ 的 key 形如 sctp{uid}t...，走 {uid}.push.ft07.com；旧版 Turbo key 走 sctapi。"""
    m = re.match(r"^sctp(\d+)t", key)
    if m:
        return f"https://{m.group(1)}.push.ft07.com/send/{key}.send"
    return f"https://sctapi.ftqq.com/{key}.send"


def to_markdown(text: str) -> str:
    return "\n".join(line + "  " if line.strip() else "" for line in text.splitlines())


def send_message(subject: str, text: str) -> bool:
    """发送一条微信推送，返回是否已送达；失败返回 False，信号留到下次运行重试该渠道。"""
    key = sendkey()
    if not key:
        log.warning("未配置 SERVERCHAN_SENDKEY，跳过微信推送")
        return True
    title = subject if len(subject) <= TITLE_LIMIT else subject[: TITLE_LIMIT - 1] + "…"
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(api_url(key), data={"title": title, "desp": to_markdown(text)},
                                 timeout=15)
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") == 0:
                return True
            # 业务错误（key 无效、超出当日额度等）重试也没用，直接报失败
            log.error("微信推送被拒: %s", body.get("message") or body)
            return False
        except (requests.RequestException, ValueError) as e:
            last_err = e
            wait = 2**attempt
            log.warning("微信推送第 %d 次失败: %s，%ds 后重试", attempt, e, wait)
            time.sleep(wait)
    log.error("微信推送失败: %s", last_err)
    return False
