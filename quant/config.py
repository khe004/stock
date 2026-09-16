"""读取 config.yaml 与 .env，提供全局配置对象。"""

from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, raw: dict):
        self.raw = raw
        self._universe_cache: dict[str, list[str]] = {}

    @property
    def db_path(self) -> Path:
        return ROOT / self.raw["database"]

    @property
    def history_start(self) -> str:
        return str(self.raw.get("history_start", "2015-01-01"))

    @property
    def watchlist(self) -> dict[str, list[str]]:
        return self.raw["watchlist"]

    @property
    def all_symbols(self) -> list[str]:
        seen: dict[str, None] = {}
        for symbols in self.watchlist.values():
            for s in symbols:
                seen.setdefault(s)
        return list(seen)

    @property
    def ai_infra_symbols(self) -> list[str]:
        """AI 基建观察池的全部标的（按 universe_ai_infra.yaml 去重保序）。"""
        try:
            return self.universe_symbols("universe_ai_infra.yaml")
        except (OSError, KeyError, AttributeError):
            # 兼容没有独立观察池文件的旧配置；当前配置会走上面的文件。
            return list(dict.fromkeys(self.watchlist.get("ai_infra", [])))

    @property
    def research_symbols(self) -> list[str]:
        """需要基本面/财报刷新的研究标的：S&P500 候选池 + AI 基建观察池。

        行情更新仍由 ``update_symbols`` 控制；研究数据不能只跟随策略候选池，
        否则 AI 页面里新增的池外公司会永远停留在旧快照。
        """
        seen: dict[str, None] = {}
        for s in self.universe_symbols("universe_sp500.yaml"):
            seen.setdefault(s)
        for s in self.ai_infra_symbols:
            seen.setdefault(s)
        return list(seen)

    @property
    def quarterly_research_symbols(self) -> list[str]:
        """阶段 2 首批季度三表验证样本。"""
        configured = self.raw.get("quarterly_research", {}).get("symbols", [])
        return list(dict.fromkeys(configured))

    def symbols_for(self, groups: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for g in groups:
            for s in self.watchlist.get(g, []):
                seen.setdefault(s)
        return list(seen)

    def enabled_strategies(self) -> list[tuple[str, dict]]:
        """返回启用的策略 (名称, 参数dict)，参数含 groups。"""
        out = []
        for name, params in self.raw.get("strategies", {}).items():
            if params.get("enabled", False):
                out.append((name, {k: v for k, v in params.items() if k != "enabled"}))
        return out

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.raw.get("notify", {}).get("telegram", False))

    @property
    def email_enabled(self) -> bool:
        return bool(self.raw.get("notify", {}).get("email", False))

    @property
    def cost_bps(self) -> float:
        return float(self.raw.get("backtest", {}).get("cost_bps", 0))

    @property
    def model_portfolio(self) -> list[str]:
        """推荐模型组合的全部成分（相关性页「🧺 模型组合」的默认勾选）。

        显式写进 config 而不是藏在面板代码里——这是一个有实测依据、会随结论变的决定，
        应该像策略参数一样被版本化、可追溯。空/缺省时面板回落到"全部会推送的策略"。

        成分分两类：`strategies`（本平台的策略名）+ `hold_assets`（买入持有的 ETF，
        不是策略、没有信号，作为一条独立收益腿参与组合，如管理期货 DBMF）。返回二者合并。
        """
        mp = self.raw.get("model_portfolio", {})
        return list(mp.get("strategies", [])) + list(mp.get("hold_assets", []))

    @property
    def model_portfolio_hold_assets(self) -> list[str]:
        """模型组合里的买入持有成分（ETF 代码），需在相关性页把它们的日收益率
        作为独立列接进 returns_df 才能被组合识别。"""
        return list(self.raw.get("model_portfolio", {}).get("hold_assets", []))

    def universe_symbols(self, filename: str) -> list[str]:
        """读取按行业分组的候选超集文件，返回全部代码（去重保序）。"""
        if filename not in self._universe_cache:
            with open(ROOT / filename, encoding="utf-8") as f:
                grouped = yaml.safe_load(f)
            seen: dict[str, None] = {}
            for syms in grouped.values():
                for s in syms:
                    seen.setdefault(s)
            self._universe_cache[filename] = list(seen)
        return self._universe_cache[filename]

    @property
    def update_symbols(self) -> list[str]:
        """每日需要更新行情的全部代码：watchlist + 各策略的候选超集。"""
        seen: dict[str, None] = dict.fromkeys(self.all_symbols)
        for _, params in self.enabled_strategies():
            if params.get("universe_file"):
                for s in self.universe_symbols(params["universe_file"]):
                    seen.setdefault(s)
        return list(seen)


def load_config(path: Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    path = path or ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
