"""AppTest 冒烟验证 AI 基建页能渲染、不报异常。"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest
import pandas as pd

from quant.data import store


def _seed_partial_db(path: Path) -> None:
    conn = store.connect(path)
    dates = pd.bdate_range("2025-01-02", periods=270)
    values = pd.Series(range(100, 370), dtype=float).to_numpy()
    frame = pd.DataFrame({
        "open": values, "high": values + 1,
        "low": values - 1, "close": values,
        "adj_close": values, "volume": 1_000_000,
    }, index=dates)
    store.upsert_prices(conn, "NVDA", frame)
    store.upsert_fundamentals(
        conn, "NVDA", "2026-09-15", "2026-09-15T23:00:00+00:00",
        {"market_cap": 3e12, "forward_pe": 30.0, "ev_to_ebitda": 25.0,
         "price_to_sales": 20.0, "revenue_growth": 0.5},
        {"currency": "USD", "financialCurrency": "USD", "shortName": "NVIDIA"},
    )
    conn.close()


def _seed_multicurrency_db(path: Path) -> None:
    _seed_partial_db(path)
    conn = store.connect(path)
    dates = pd.bdate_range("2025-01-02", periods=270)
    prices = pd.Series(range(50000, 50270), dtype=float).to_numpy()
    frame = pd.DataFrame({
        "open": prices, "high": prices + 100, "low": prices - 100,
        "close": prices, "adj_close": prices, "volume": 2_000_000,
    }, index=dates)
    store.upsert_prices(conn, "005930.KS", frame)
    fx = frame.copy()
    for col in ("open", "high", "low", "close", "adj_close"):
        fx[col] = 1400.0
    store.upsert_prices(conn, "KRW=X", fx)
    store.upsert_fundamentals(
        conn, "005930.KS", "2026-09-15", "2026-09-15T23:00:00+00:00",
        {"market_cap": 1.5e15, "forward_pe": 12.0, "ev_to_ebitda": 8.0,
         "price_to_sales": 3.0, "revenue_growth": 0.2},
        {"currency": "KRW", "financialCurrency": "KRW", "shortName": "Samsung"},
    )
    store.insert_financial_snapshot(conn, {
        "snapshot_id": uuid.uuid4().hex, "symbol": "005930.KS",
        "frequency": "quarterly", "currency": "KRW", "trading_currency": "KRW",
        "source": "test", "source_url": "https://example.test/samsung",
        "published_at": "2026-07-30", "captured_at": "2026-09-15T23:00:00+00:00",
    }, [{
        "period_start": "2026-04-01", "period_end": "2026-06-30",
        "statement": "income", "metric": "revenue", "value": 171.5e12,
        "unit": "KRW", "raw_label": "Revenue",
    }])
    conn.close()


def test_ai_infra_page_renders_with_isolated_empty_db(tmp_path, monkeypatch):
    """空的隔离数据库应显示缺数据提示，不能读取用户真实数据库。"""
    monkeypatch.setenv("QUANT_DB_PATH", str(tmp_path / "empty.db"))
    at = AppTest.from_file("quant/web/app.py", default_timeout=30)
    at.run()
    # 切换到 AI 基建页面
    pills = at.pills
    assert len(pills) > 0, "页面导航 pills 不存在"
    # 找到 AI 基建页面
    pills[0].set_value("🤖 AI 基建").run()
    assert not at.exception, f"AI 基建页面渲染报异常: {at.exception}"
    assert any("没有 AI 基建标的行情数据" in item.value for item in at.warning)


def test_ai_infra_partial_data_components_and_lane_switch(tmp_path, monkeypatch):
    """部分数据场景仍应渲染状态、赛道表和公司详情。"""
    db = tmp_path / "partial.db"
    _seed_partial_db(db)
    monkeypatch.setenv("QUANT_DB_PATH", str(db))
    at = AppTest.from_file("quant/web/app.py", default_timeout=30)
    at.run()
    pills = at.pills
    pills[0].set_value("🤖 AI 基建").run()
    assert not at.exception, f"初始渲染报异常: {at.exception}"
    assert any(item.value == "研究数据状态" for item in at.subheader)
    assert any(item.value == "赛道概览" for item in at.subheader)
    sort = next(item for item in at.selectbox if item.key == "ai_infra_sort")
    sort.set_value("近1月").run()
    assert next(item for item in at.selectbox if item.key == "ai_infra_sort").value == "近1月"
    # 尝试切换到另一个赛道
    selectboxes = at.selectbox
    if selectboxes:
        # 找到 ai_infra_lane selectbox
        for sb in selectboxes:
            if sb.key == "ai_infra_lane":
                # 切换到存储赛道
                sb.set_value("存储").run()
                assert not at.exception, f"切换赛道报异常: {at.exception}"
                assert next(item for item in at.selectbox
                            if item.key == "ai_infra_sort").value == "近1月"
                break


def test_ai_infra_multicurrency_detail_uses_statement_currency(tmp_path, monkeypatch):
    db = tmp_path / "multicurrency.db"
    _seed_multicurrency_db(db)
    monkeypatch.setenv("QUANT_DB_PATH", str(db))
    at = AppTest.from_file("quant/web/app.py", default_timeout=30)
    at.run()
    at.pills[0].set_value("🤖 AI 基建").run()
    lane = next(item for item in at.selectbox if item.key == "ai_infra_lane")
    lane.set_value("存储").run()
    company = next(item for item in at.selectbox if item.key == "ai_infra_symbol")
    company.set_value("005930.KS").run()
    assert not at.exception
    assert any("财报币种：KRW" in item.value and "交易币种：KRW" in item.value
               for item in at.caption)
