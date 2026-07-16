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
