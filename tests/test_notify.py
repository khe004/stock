import smtplib

import pytest

from quant.notify.email import send_email

ENV = {
    "SMTP_HOST": "smtp.test",
    "SMTP_PORT": "587",
    "SMTP_USER": "me@test.com",
    "SMTP_PASSWORD": "pw",
    "EMAIL_TO": "a@x.com, b@y.com",
}


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.sent = None
        self.logged_in = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ehlo(self):
        pass

    def has_extn(self, name):
        return False

    def login(self, user, password):
        self.logged_in = (user, password)

    def sendmail(self, from_addr, to_addrs, body):
        self.sent = (from_addr, to_addrs, body)


def set_env(monkeypatch, **overrides):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in {**ENV, **overrides}.items():
        if v is not None:
            monkeypatch.setenv(k, v)


def test_send_email_success(monkeypatch):
    set_env(monkeypatch)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    FakeSMTP.instances.clear()
    assert send_email("测试主题", "测试正文") is True
    inst = FakeSMTP.instances[0]
    from_addr, to_addrs, body = inst.sent
    assert from_addr == "me@test.com"
    assert to_addrs == ["a@x.com", "b@y.com"]
    assert inst.logged_in == ("me@test.com", "pw")
    assert "=?utf-8?" in body  # 中文主题按 UTF-8 编码


def test_unconfigured_is_noop(monkeypatch):
    set_env(monkeypatch, SMTP_HOST=None, SMTP_USER=None, EMAIL_TO=None)

    def boom(*a, **kw):
        raise AssertionError("未配置时不应尝试连接")

    monkeypatch.setattr(smtplib, "SMTP", boom)
    assert send_email("s", "t") is True


def test_smtp_failure_returns_false(monkeypatch):
    set_env(monkeypatch)

    def broken(*a, **kw):
        raise smtplib.SMTPException("connection refused")

    monkeypatch.setattr(smtplib, "SMTP", broken)
    assert send_email("s", "t") is False


def test_ssl_port_uses_smtp_ssl(monkeypatch):
    set_env(monkeypatch, SMTP_PORT="465")
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)

    def boom(*a, **kw):
        raise AssertionError("465 端口应使用 SMTP_SSL")

    monkeypatch.setattr(smtplib, "SMTP", boom)
    FakeSMTP.instances.clear()
    assert send_email("s", "t") is True
    assert FakeSMTP.instances[0].sent is not None


def test_missing_recipient_is_noop(monkeypatch):
    set_env(monkeypatch, EMAIL_TO="  ")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    FakeSMTP.instances.clear()
    assert send_email("s", "t") is True
    assert FakeSMTP.instances == []


def test_wechat_api_url_supports_both_key_formats():
    from quant.notify import wechat
    assert wechat.api_url("sctp12345tABCDEF") == "https://12345.push.ft07.com/send/sctp12345tABCDEF.send"
    assert wechat.api_url("SCT999XYZ") == "https://sctapi.ftqq.com/SCT999XYZ.send"


def test_wechat_markdown_keeps_line_breaks_and_title_is_truncated(monkeypatch):
    from quant.notify import wechat
    sent = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"code": 0}

    def fake_post(url, data, timeout):
        sent.update(data); return Resp()

    monkeypatch.setenv("SERVERCHAN_SENDKEY", "SCTkey")
    monkeypatch.setattr(wechat.requests, "post", fake_post)
    assert wechat.send_message("标" * 40, "第一行\n第二行\n\n第三行")
    assert len(sent["title"]) == 32 and sent["title"].endswith("…")
    assert sent["desp"] == "第一行  \n第二行  \n\n第三行  "


def test_wechat_business_error_is_not_delivered(monkeypatch):
    from quant.notify import wechat

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"code": 40001, "message": "bad key"}

    monkeypatch.setenv("SERVERCHAN_SENDKEY", "SCTkey")
    monkeypatch.setattr(wechat.requests, "post", lambda *a, **k: Resp())
    assert wechat.send_message("t", "x") is False


def test_wechat_channel_needs_both_config_and_key(monkeypatch):
    from quant.config import load_config
    monkeypatch.setenv("SERVERCHAN_SENDKEY", "")
    assert not load_config().wechat_enabled


def test_retired_symbols_skip_fetch_but_stay_out_of_strategies():
    import run_daily
    from quant.config import load_config
    cfg = load_config()
    retired = cfg.retired_symbols
    assert {"EA", "EQR"} <= retired
    assert not retired & set(cfg.update_symbols)
    fed = {s for _, p in cfg.enabled_strategies() for s in run_daily._strategy_symbols(cfg, p)}
    assert not retired & fed
    # 名单文件不删：回测仍可用它们的历史
    assert "EA" in cfg.universe_symbols("universe_sp500.yaml")
