"""SQLite 存储：行情表 prices、信号表 signals。"""

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


# 迁移用：两张表的期望列。老版本库（早期在用户机器上重建过的 schema）可能缺列
PRICES_COL_TYPES = {
    "open": "REAL", "high": "REAL", "low": "REAL",
    "close": "REAL", "adj_close": "REAL", "volume": "INTEGER",
}
SIGNALS_COLS = {"id", "date", "symbol", "strategy", "direction",
                "price", "strength", "reason", "created_at", "notified_at"}


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
    conn.commit()


def connect(db_path: Path | str) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
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
