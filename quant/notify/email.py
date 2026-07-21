"""SMTP 邮件通知。凭据与收件人全部来自环境变量（.env），未配置时视为 noop。"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

log = logging.getLogger(__name__)


def send_email(subject: str, text: str) -> bool:
    """发送纯文本邮件，返回“是否已送达”。未配置返回 True（noop，避免信号
    永远滞留在未通知状态）；发送失败返回 False，下次运行重试。"""
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD", "")
    to = [a.strip() for a in os.getenv("EMAIL_TO", "").split(",") if a.strip()]
    port = int(os.getenv("SMTP_PORT", "587"))
    if not host or not user or not to:
        log.warning("未配置 SMTP_HOST/SMTP_USER/EMAIL_TO，跳过邮件通知")
        return True

    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("投资信号", user))
    msg["To"] = ", ".join(to)

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=30)
        else:
            server = smtplib.SMTP(host, port, timeout=30)
        with server:
            server.ehlo()
            # 服务器支持才升级 TLS（Gmail 等公网服务都支持；本地调试服务器可不加密）
            if server.has_extn("starttls"):
                server.starttls()
                server.ehlo()
            if password:
                server.login(user, password)
            server.sendmail(user, to, msg.as_string())
        log.info("邮件已发送至 %s", ", ".join(to))
        return True
    except (smtplib.SMTPException, OSError) as e:
        log.error("邮件发送失败: %s", e)
        return False
