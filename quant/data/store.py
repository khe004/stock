"""SQLite 存储：行情表 prices、信号表 signals。"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant.strategies.base import Signal

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


def connect(db_path: Path | str) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
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
