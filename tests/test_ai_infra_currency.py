"""多币种换算与币种推断的单元测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


class TestInferCurrency:
    """币种推断（按交易所后缀）。"""

    def test_korean_ks(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("005930.KS") == "KRW"

    def test_korean_kq(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("373220.KQ") == "KRW"

    def test_japanese(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("8035.T") == "JPY"

    def test_taiwan(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("2317.TW") == "TWD"

    def test_shenzhen(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("300308.SZ") == "CNY"

    def test_shanghai(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("600519.SS") == "CNY"

    def test_hong_kong(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("0700.HK") == "HKD"

    def test_us_no_suffix(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("NVDA") == "USD"

    def test_us_special_chars(self):
        from quant.analysis.ai_infra import infer_currency
        assert infer_currency("BRK-B") == "USD"


class TestGetCurrencyForSymbol:
    """币种获取（优先 raw_json，兜底后缀推断）。"""

    def test_from_raw_json(self):
        from quant.analysis.ai_infra import get_currency_for_symbol
        raw = {"currency": "KRW", "shortName": "Samsung"}
        assert get_currency_for_symbol("005930.KS", raw) == "KRW"

    def test_raw_json_overrides_suffix(self):
        """raw_json 里的 currency 优先于后缀推断。"""
        from quant.analysis.ai_infra import get_currency_for_symbol
        # 假设一个标的后缀是 .HK 但实际以 USD 交易
        raw = {"currency": "USD"}
        assert get_currency_for_symbol("9988.HK", raw) == "USD"

    def test_fallback_to_suffix(self):
        from quant.analysis.ai_infra import get_currency_for_symbol
        assert get_currency_for_symbol("8035.T", None) == "JPY"

    def test_fallback_empty_raw(self):
        from quant.analysis.ai_infra import get_currency_for_symbol
        assert get_currency_for_symbol("2317.TW", {}) == "TWD"

    def test_raw_json_missing_currency_key(self):
        from quant.analysis.ai_infra import get_currency_for_symbol
        raw = {"shortName": "Samsung"}
        assert get_currency_for_symbol("005930.KS", raw) == "KRW"

    def test_case_insensitive_raw(self):
        """raw_json 里的 currency 应被 upper() 处理。"""
        from quant.analysis.ai_infra import get_currency_for_symbol
        raw = {"currency": "jpy"}
        assert get_currency_for_symbol("8035.T", raw) == "JPY"


class TestToUsdMarketCap:
    """本币市值 → 美元市值换算。"""

    def test_usd_passthrough(self):
        """USD 标的直接返回原值。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        caps = {"NVDA": 3_000_000_000_000.0}
        currencies = {"NVDA": "USD"}
        fx = {"KRW": 1419.0}
        result = to_usd_market_cap(caps, currencies, fx)
        assert result["NVDA"] == 3_000_000_000_000.0

    def test_krw_conversion(self):
        """韩元市值正确换算成美元。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        # 三星电子本币市值 ~1530万亿韩元，1 USD = 1419 KRW
        caps = {"005930.KS": 1_530_000_000_000_000.0}
        currencies = {"005930.KS": "KRW"}
        fx = {"KRW": 1419.0}
        result = to_usd_market_cap(caps, currencies, fx)
        expected = 1_530_000_000_000_000.0 / 1419.0
        assert result["005930.KS"] == pytest.approx(expected, rel=1e-6)
        # 应该约 $1078B 量级，不是万亿美元级
        assert result["005930.KS"] < 2_000_000_000_000  # < $2T

    def test_missing_fx_returns_none(self):
        """缺汇率时必须返回 None，不能返回原值。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        caps = {"005930.KS": 1_530_000_000_000_000.0}
        currencies = {"005930.KS": "KRW"}
        fx = {}  # 没有 KRW 汇率
        result = to_usd_market_cap(caps, currencies, fx)
        assert result["005930.KS"] is None

    def test_none_cap_stays_none(self):
        """原始市值为 None 时结果也为 None。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        caps = {"NVDA": None}
        currencies = {"NVDA": "USD"}
        fx = {}
        result = to_usd_market_cap(caps, currencies, fx)
        assert result["NVDA"] is None

    def test_empty_input(self):
        """空输入返回空 dict。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        result = to_usd_market_cap({}, {}, {})
        assert result == {}

    def test_mixed_currencies(self):
        """混合币种全部正确换算。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        caps = {
            "NVDA": 3_000e9,
            "005930.KS": 1_530e12,  # 韩元
            "8035.T": 25e12,        # 日元
        }
        currencies = {
            "NVDA": "USD",
            "005930.KS": "KRW",
            "8035.T": "JPY",
        }
        fx = {"KRW": 1419.0, "JPY": 158.0}
        result = to_usd_market_cap(caps, currencies, fx)
        assert result["NVDA"] == 3_000e9
        assert result["005930.KS"] == pytest.approx(1_530e12 / 1419.0, rel=1e-6)
        assert result["8035.T"] == pytest.approx(25e12 / 158.0, rel=1e-6)

    def test_zero_fx_rate_returns_none(self):
        """汇率为 0 时返回 None（避免除以零）。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        caps = {"005930.KS": 1_530e12}
        currencies = {"005930.KS": "KRW"}
        fx = {"KRW": 0.0}
        result = to_usd_market_cap(caps, currencies, fx)
        assert result["005930.KS"] is None

    def test_default_currency_is_usd(self):
        """currencies dict 里没有该 symbol 时默认当 USD。"""
        from quant.analysis.ai_infra import to_usd_market_cap
        caps = {"NVDA": 3_000e9}
        currencies = {}  # 没有 NVDA 的币种
        fx = {}
        result = to_usd_market_cap(caps, currencies, fx)
        assert result["NVDA"] == 3_000e9


class TestLaneShareWithMixedCurrencies:
    """混合币种后赛道份额加总仍为 100%。"""

    def test_shares_sum_to_one(self):
        """换算后的美元市值份额应加总为 1.0。"""
        from quant.analysis.ai_infra import compute_lane_market_share, to_usd_market_cap
        # 模拟存储赛道（6 只）
        raw_caps = {
            "MU": 189e9,              # USD
            "WDC": 30e9,              # USD
            "STX": 25e9,              # USD
            "SNDK": 20e9,             # USD
            "005930.KS": 1_530e12,    # KRW
            "000660.KS": 1_018e12,    # KRW
        }
        currencies = {
            "MU": "USD", "WDC": "USD", "STX": "USD", "SNDK": "USD",
            "005930.KS": "KRW", "000660.KS": "KRW",
        }
        fx = {"KRW": 1419.0}
        usd_caps = to_usd_market_cap(raw_caps, currencies, fx)
        lane = list(raw_caps.keys())
        shares = compute_lane_market_share(lane, usd_caps)
        total = sum(v for v in shares.values() if v is not None)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_mu_share_drops_with_samsung(self):
        """加入三星后，MU 份额应从 ~64% 降到 ~30% 范围。"""
        from quant.analysis.ai_infra import compute_lane_market_share, to_usd_market_cap
        raw_caps = {
            "MU": 189e9,
            "WDC": 30e9,
            "STX": 25e9,
            "SNDK": 20e9,
            "005930.KS": 1_530e12,
            "000660.KS": 1_018e12,
        }
        currencies = {
            "MU": "USD", "WDC": "USD", "STX": "USD", "SNDK": "USD",
            "005930.KS": "KRW", "000660.KS": "KRW",
        }
        fx = {"KRW": 1419.0}
        usd_caps = to_usd_market_cap(raw_caps, currencies, fx)
        lane = list(raw_caps.keys())
        shares = compute_lane_market_share(lane, usd_caps)
        # MU should be around 10-15% (189B / ~2000B)
        assert shares["MU"] is not None
        assert shares["MU"] < 0.40  # 远低于之前的 64%


# ── display_name：非美代码可读性 ──────────────────────────────

def test_display_name_prefers_chinese_map():
    from quant.analysis.ai_infra import display_name
    # 纯数字代码必须给出中文名，否则页面上完全无法辨认
    assert display_name("005930.KS") == "三星电子"
    assert display_name("300308.SZ") == "中际旭创"
    assert display_name("TSM") == "台积电"
    # 中文名表优先于 yfinance 英文名
    assert display_name("2317.TW", {"shortName": "HON HAI PRECISION"}) == "鸿海精密"


def test_display_name_falls_back_to_yfinance_then_symbol():
    from quant.analysis.ai_infra import display_name
    # 不在中文表里 → 用 shortName
    assert display_name("NVDA", {"shortName": "NVIDIA Corporation"}) == "NVIDIA Corporation"
    # 无 shortName → longName
    assert display_name("XYZ", {"longName": "Xyz Holdings Inc."}) == "Xyz Holdings Inc."
    # 都没有 → 退回代码本身，不返回 None/空
    assert display_name("XYZ", {}) == "XYZ"
    assert display_name("XYZ", None) == "XYZ"
    # 空字符串不算有效名字
    assert display_name("XYZ", {"shortName": "", "longName": "  "}) == "XYZ"
