"""SQLite 存储：行情表 prices、信号表 signals、基本面表 fundamentals。"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant.strategies.base import Signal

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol     TEXT NOT NULL,
    date       TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    adj_close  REAL,
    volume     INTEGER,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    direction   TEXT NOT NULL,
    price       REAL,
    strength    REAL,
    reason      TEXT,
    created_at  TEXT NOT NULL,
    notified_at TEXT,
    UNIQUE (date, symbol, strategy, direction)
);
"""

SCHEMA_FUNDAMENTALS = """
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    trailing_pe     REAL,
    forward_pe      REAL,
    price_to_book   REAL,
    price_to_sales  REAL,
    ev_to_ebitda    REAL,
    peg_ratio       REAL,
    dividend_yield  REAL,
    trailing_eps    REAL,
    return_on_equity REAL,
    profit_margins  REAL,
    gross_margins   REAL,
    debt_to_equity  REAL,
    market_cap      REAL,
    book_value      REAL,
    beta            REAL,
    raw_json        TEXT,
    PRIMARY KEY (symbol, date)
);
"""


# 迁移用：两张表的期望列。老版本库（早期在用户机器上重建过的 schema）可能缺列
PRICES_COL_TYPES = {
    "open": "REAL", "high": "REAL", "low": "REAL",
    "close": "REAL", "adj_close": "REAL", "volume": "INTEGER",
}
SIGNALS_COLS = {"id", "date", "symbol", "strategy", "direction",
                "price", "strength", "reason", "created_at", "notified_at"}
FUNDAMENTALS_COLS = {
    "symbol", "date", "captured_at",
    "trailing_pe", "forward_pe", "price_to_book", "price_to_sales",
    "ev_to_ebitda", "peg_ratio", "dividend_yield", "trailing_eps",
    "return_on_equity", "profit_margins", "gross_margins", "debt_to_equity",
    "market_cap", "book_value", "beta", "raw_json",
}


def _table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    """旧版数据库自愈：prices 缺列就补列（行情数据宝贵，保留），
    signals 结构不符则重建（信号可由策略随时重算，旧表保留备份）。"""
    missing = set(PRICES_COL_TYPES) - _table_cols(conn, "prices")
    for col in sorted(missing):
        conn.execute(f"ALTER TABLE prices ADD COLUMN {col} {PRICES_COL_TYPES[col]}")
        log.warning("prices 表缺列 %s，已补加（值为空）；建议运行 --full-refresh 回填", col)
    if not SIGNALS_COLS <= _table_cols(conn, "signals"):
        conn.execute("DROP TABLE IF EXISTS signals_legacy")
        conn.execute("ALTER TABLE signals RENAME TO signals_legacy")
        conn.executescript(SCHEMA)
        log.warning("signals 表结构过旧，已重建（旧表保留为 signals_legacy）；"
                    "历史信号可用 run_daily.py --date 补跑重算")
    # fundamentals 表：数据宝贵不 DROP，只补缺列
    if "fundamentals" not in {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}:
        conn.executescript(SCHEMA_FUNDAMENTALS)
        log.info("fundamentals 表不存在，已自动创建")
    conn.commit()


def connect(db_path: Path | str) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executescript(SCHEMA_FUNDAMENTALS)
    _migrate(conn)
    return conn


def upsert_prices(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    """写入行情，重复 (symbol, date) 覆盖。df 需含 open/high/low/close/adj_close/volume，索引为日期。"""
    rows = [
        (
            symbol,
            idx.strftime("%Y-%m-%d"),
            row.get("open"), row.get("high"), row.get("low"),
            row.get("close"), row.get("adj_close"),
            int(row["volume"]) if pd.notna(row.get("volume")) else None,
        )
        for idx, row in df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return len(rows)


def latest_price_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(date) AS d FROM prices WHERE symbol = ?", (symbol,)
    ).fetchone()
    return row["d"]


def load_prices(conn: sqlite3.Connection, symbol: str, start: str | None = None) -> pd.DataFrame:
    """按日期升序返回某标的行情，DatetimeIndex。"""
    query = "SELECT date, open, high, low, close, adj_close, volume FROM prices WHERE symbol = ?"
    params: list = [symbol]
    if start:
        query += " AND date >= ?"
        params.append(start)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, index_col="date", parse_dates=["date"])
    return df


def insert_signals(conn: sqlite3.Connection, signals: list[Signal]) -> int:
    """写入信号，(date, symbol, strategy, direction) 已存在则忽略。返回新插入条数。"""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = 0
    for s in signals:
        cur = conn.execute(
            """INSERT OR IGNORE INTO signals
               (date, symbol, strategy, direction, price, strength, reason, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (s.date, s.symbol, s.strategy, s.direction, s.price, s.strength, s.reason, now),
        )
        new += cur.rowcount
    conn.commit()
    return new


def unnotified_signals(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM signals WHERE notified_at IS NULL ORDER BY date, symbol"
    ).fetchall()


def mark_all_notified(conn: sqlite3.Connection) -> int:
    """把所有未通知信号标记为已通知（backfill 用：历史信号只补记录不推送）。"""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "UPDATE signals SET notified_at = ? WHERE notified_at IS NULL", (now,)
    )
    conn.commit()
    return cur.rowcount


def mark_notified(conn: sqlite3.Connection, ids: list[int]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany("UPDATE signals SET notified_at = ? WHERE id = ?", [(now, i) for i in ids])
    conn.commit()


def load_signals(
    conn: sqlite3.Connection,
    strategy: str | None = None,
    symbol: str | None = None,
    start: str | None = None,
) -> pd.DataFrame:
    query = "SELECT date, symbol, strategy, direction, price, strength, reason, notified_at FROM signals WHERE 1=1"
    params: list = []
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if start:
        query += " AND date >= ?"
        params.append(start)
    query += " ORDER BY date DESC, symbol"
    return pd.read_sql_query(query, conn, params=params)


def upsert_fundamentals(conn: sqlite3.Connection, symbol: str, date: str,
                        captured_at: str, metrics: dict, raw: dict) -> int:
    """写入基本面快照，(symbol, date) 已存在则覆盖。返回 1。"""
    conn.execute(
        """INSERT OR REPLACE INTO fundamentals
           (symbol, date, captured_at,
            trailing_pe, forward_pe, price_to_book, price_to_sales,
            ev_to_ebitda, peg_ratio, dividend_yield, trailing_eps,
            return_on_equity, profit_margins, gross_margins, debt_to_equity,
            market_cap, book_value, beta, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            symbol, date, captured_at,
            metrics.get("trailing_pe"), metrics.get("forward_pe"),
            metrics.get("price_to_book"), metrics.get("price_to_sales"),
            metrics.get("ev_to_ebitda"), metrics.get("peg_ratio"),
            metrics.get("dividend_yield"), metrics.get("trailing_eps"),
            metrics.get("return_on_equity"), metrics.get("profit_margins"),
            metrics.get("gross_margins"), metrics.get("debt_to_equity"),
            metrics.get("market_cap"), metrics.get("book_value"),
            metrics.get("beta"),
            json.dumps(raw, ensure_ascii=False) if raw else None,
        ),
    )
    conn.commit()
    return 1


def latest_fundamentals_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    """返回该标的最新一条基本面快照的日期，无记录返回 None。"""
    row = conn.execute(
        "SELECT MAX(date) AS d FROM fundamentals WHERE symbol = ?", (symbol,)
    ).fetchone()
    return row["d"]


def load_fundamentals(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    start: str | None = None,
) -> pd.DataFrame:
    """按日期升序返回基本面快照（含 raw_json）。"""
    query = "SELECT * FROM fundamentals WHERE 1=1"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if start:
        query += " AND date >= ?"
        params.append(start)
    query += " ORDER BY symbol, date"
    return pd.read_sql_query(query, conn, params=params)
