"""yfinance 日线数据拉取：增量更新 + 失败重试。"""

import logging
import time
import pandas as pd
import yfinance as yf

from quant.data import store

log = logging.getLogger(__name__)

MAX_RETRIES = 3

COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=COLUMN_MAP)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def fetch_history(symbol: str, start: str) -> pd.DataFrame:
    """拉取 start 至今的日线，带指数退避重试。失败抛 RuntimeError。

    注意：yfinance 被限流时经常不抛异常而是静默返回空表，因此空表也按
    可重试处理；重试用尽仍为空才返回空表（真退市的代码就是这种表现）。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        df = None
        try:
            df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
        except Exception as e:  # noqa: BLE001 - yfinance 抛的异常类型不稳定
            last_err = e
        if df is not None and not df.empty:
            return _normalize(df)
        if attempt < MAX_RETRIES:
            wait = 2**attempt
            log.warning("%s 第 %d 次拉取%s，%ds 后重试", symbol, attempt,
                        f"失败: {last_err}" if last_err else "返回空表（疑似限流）", wait)
            time.sleep(wait)
    if last_err is not None:
        raise RuntimeError(f"{symbol}: 拉取失败（重试 {MAX_RETRIES} 次）: {last_err}")
    return pd.DataFrame()


def update_symbol(conn, symbol: str, history_start: str, full: bool = False) -> int:
    """增量更新单个标的，返回写入行数。

    full=True 时忽略库内进度、从 history_start 全量重拉并覆盖：yfinance 的
    adj_close 以下载日为基准回溯复权，分红后历史行会整体变化，增量拼接会在
    衔接点留下微小错位，建议每季度全量刷新一次。

    增量起点是库内最新日期"本身"而非其后一天：盘中运行会存下当日的半根K线，
    从最新日期重拉可保证下次运行将其覆盖为收盘定稿（REPLACE 幂等）。"""
    latest = None if full else store.latest_price_date(conn, symbol)
    start = latest if latest else history_start
    df = fetch_history(symbol, start)
    if df.empty:
        if latest is None:
            # yfinance 拉取失败时常静默返回空表；首拉/全量拿到空必属异常
            raise RuntimeError(f"{symbol}: 拉取返回空数据（网络受限或代码无效？）")
        return 0
    return store.upsert_prices(conn, symbol, df)


def update_all(conn, symbols: list[str], history_start: str,
               full: bool = False) -> tuple[int, list[str]]:
    """更新全部标的。返回 (总写入行数, 失败标的列表)。"""
    total, failed = 0, []
    for symbol in symbols:
        try:
            n = update_symbol(conn, symbol, history_start, full=full)
            log.info("%s 更新 %d 行", symbol, n)
            total += n
        except Exception as e:  # noqa: BLE001
            log.error("%s 更新失败: %s", symbol, e)
            failed.append(symbol)
    return total, failed
