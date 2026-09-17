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
    revenue_growth  REAL,
    earnings_growth REAL,
    raw_json        TEXT,
    PRIMARY KEY (symbol, date)
);
"""

# fundamentals 后加的列 → 类型（旧库自愈补列用；数据宝贵不 DROP 重建）
FUNDAMENTALS_ADDED_COLS = {
    "revenue_growth": "REAL",
    "earnings_growth": "REAL",
}

SCHEMA_FINANCIALS = """
CREATE TABLE IF NOT EXISTS financials (
    symbol TEXT NOT NULL,
    fiscal_date TEXT NOT NULL,
    revenue REAL,
    gross_profit REAL,
    operating_income REAL,
    net_income REAL,
    currency TEXT,
    trading_currency TEXT,
    source TEXT,
    source_url TEXT,
    captured_at TEXT NOT NULL,
    PRIMARY KEY (symbol, fiscal_date)
);
"""

SCHEMA_RESEARCH_UPDATES = """
CREATE TABLE IF NOT EXISTS research_updates (
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT,
    PRIMARY KEY (symbol, data_type)
);
CREATE TABLE IF NOT EXISTS research_update_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_update_runs
    ON research_update_runs(symbol, data_type, checked_at);
"""

SCHEMA_FINANCIAL_FACTS = """
CREATE TABLE IF NOT EXISTS financial_statement_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    frequency TEXT NOT NULL,
    currency TEXT,
    trading_currency TEXT,
    source TEXT NOT NULL,
    source_url TEXT,
    published_at TEXT,
    captured_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_financial_snapshots_symbol
    ON financial_statement_snapshots(symbol, frequency, captured_at);

CREATE TABLE IF NOT EXISTS financial_facts (
    snapshot_id TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT NOT NULL,
    statement TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    unit TEXT NOT NULL,
    raw_label TEXT,
    PRIMARY KEY (snapshot_id, period_end, statement, metric),
    FOREIGN KEY (snapshot_id) REFERENCES financial_statement_snapshots(snapshot_id)
);
"""

SCHEMA_RESEARCH_RECORDS = """
CREATE TABLE IF NOT EXISTS research_note_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,
    thesis TEXT,
    assumptions TEXT,
    risks TEXT,
    next_check TEXT,
    source_url TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_notes_symbol
    ON research_note_versions(symbol, created_at);

CREATE TABLE IF NOT EXISTS business_evidence_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value_text TEXT,
    period TEXT,
    unit TEXT,
    published_at TEXT,
    source_url TEXT NOT NULL,
    excerpt TEXT,
    entry_method TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_business_evidence_symbol
    ON business_evidence_versions(symbol, created_at);

CREATE TABLE IF NOT EXISTS research_view_state (
    symbol TEXT PRIMARY KEY,
    last_viewed_at TEXT NOT NULL
);
"""

SCHEMA_NOTIFICATION_DELIVERIES = """
CREATE TABLE IF NOT EXISTS notification_deliveries (
    signal_id INTEGER NOT NULL,
    channel TEXT NOT NULL,
    delivered_at TEXT NOT NULL,
    PRIMARY KEY (signal_id, channel),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
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
FINANCIAL_SNAPSHOT_ADDED_COLS = {
    "trading_currency": "TEXT",
}
ANNUAL_FINANCIAL_ADDED_COLS = {
    "currency": "TEXT", "trading_currency": "TEXT",
    "source": "TEXT", "source_url": "TEXT",
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
    else:
        fund_missing = set(FUNDAMENTALS_ADDED_COLS) - _table_cols(conn, "fundamentals")
        for col in sorted(fund_missing):
            conn.execute(
                f"ALTER TABLE fundamentals ADD COLUMN {col} {FUNDAMENTALS_ADDED_COLS[col]}")
            log.warning("fundamentals 表缺列 %s，已补加（值为空，可从 raw_json 回填）", col)
    # 财报事实的币种与证券交易币种必须分开。ADR（例如 TSM）以 USD 交易，
    # 但其合并财报仍以 TWD 报告；混为一个字段会把金额放大约 30 倍。
    snapshot_missing = (set(FINANCIAL_SNAPSHOT_ADDED_COLS)
                        - _table_cols(conn, "financial_statement_snapshots"))
    for col in sorted(snapshot_missing):
        conn.execute(
            f"ALTER TABLE financial_statement_snapshots ADD COLUMN "
            f"{col} {FINANCIAL_SNAPSHOT_ADDED_COLS[col]}")
        log.warning("financial_statement_snapshots 表缺列 %s，已补加", col)
    annual_missing = set(ANNUAL_FINANCIAL_ADDED_COLS) - _table_cols(conn, "financials")
    for col in sorted(annual_missing):
        conn.execute(f"ALTER TABLE financials ADD COLUMN {col} "
                     f"{ANNUAL_FINANCIAL_ADDED_COLS[col]}")
        log.warning("financials 表缺列 %s，已补加", col)
    conn.commit()


def connect(db_path: Path | str) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executescript(SCHEMA_FUNDAMENTALS)
    conn.executescript(SCHEMA_FINANCIALS)
    conn.executescript(SCHEMA_RESEARCH_UPDATES)
    conn.executescript(SCHEMA_FINANCIAL_FACTS)
    conn.executescript(SCHEMA_RESEARCH_RECORDS)
    conn.executescript(SCHEMA_NOTIFICATION_DELIVERIES)
    _migrate(conn)
    return conn


def save_research_note(conn: sqlite3.Connection, symbol: str, status: str,
                       thesis: str = "", assumptions: str = "", risks: str = "",
                       next_check: str = "", source_url: str = "") -> int:
    """新增一个研究判断版本；从不覆盖历史版本。"""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cursor = conn.execute(
        """INSERT INTO research_note_versions
           (symbol, status, thesis, assumptions, risks, next_check, source_url, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, status, thesis, assumptions, risks, next_check, source_url, created_at),
    )
    conn.commit()
    return int(cursor.lastrowid)


def load_research_notes(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM research_note_versions WHERE symbol = ? ORDER BY id DESC",
        conn, params=[symbol])


def load_latest_research_notes(conn: sqlite3.Connection) -> pd.DataFrame:
    """每家公司只返回最新研究判断版本。"""
    return pd.read_sql_query(
        """SELECT n.* FROM research_note_versions AS n
           JOIN (SELECT symbol, MAX(id) AS id FROM research_note_versions GROUP BY symbol) latest
             ON latest.id = n.id
           ORDER BY n.status, n.symbol""", conn)


def save_business_evidence(conn: sqlite3.Connection, symbol: str, metric_name: str,
                           source_url: str, value_text: str = "", period: str = "",
                           unit: str = "", published_at: str = "", excerpt: str = "",
                           entry_method: str = "人工", verification_status: str = "待核验") -> int:
    """新增一条业务证据版本；source_url 必填以保证可追溯。"""
    if not source_url.strip():
        raise ValueError("业务证据必须提供来源链接")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cursor = conn.execute(
        """INSERT INTO business_evidence_versions
           (symbol, metric_name, value_text, period, unit, published_at, source_url,
            excerpt, entry_method, verification_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, metric_name, value_text, period, unit, published_at, source_url,
         excerpt, entry_method, verification_status, created_at),
    )
    conn.commit()
    return int(cursor.lastrowid)


def load_business_evidence(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM business_evidence_versions WHERE symbol = ? ORDER BY id DESC",
        conn, params=[symbol])


def mark_research_viewed(conn: sqlite3.Connection, symbol: str,
                         viewed_at: str | None = None) -> str:
    viewed_at = viewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO research_view_state(symbol, last_viewed_at) VALUES (?, ?)
           ON CONFLICT(symbol) DO UPDATE SET last_viewed_at = excluded.last_viewed_at""",
        (symbol, viewed_at),
    )
    conn.commit()
    return viewed_at


def research_changes_since_view(conn: sqlite3.Connection, symbol: str) -> list[dict]:
    """返回上次查看后新增的数据/证据；查询派生事件天然不会重复入库。"""
    row = conn.execute(
        "SELECT last_viewed_at FROM research_view_state WHERE symbol = ?", (symbol,)
    ).fetchone()
    since = row["last_viewed_at"] if row else None
    changes: list[dict] = []
    sources = [
        ("基本面快照", "SELECT MAX(captured_at) d FROM fundamentals WHERE symbol = ?"),
        ("年度财报", "SELECT MAX(captured_at) d FROM financials WHERE symbol = ?"),
        ("季度三表", "SELECT MAX(captured_at) d FROM financial_statement_snapshots WHERE symbol = ?"),
        ("业务证据", "SELECT MAX(created_at) d FROM business_evidence_versions WHERE symbol = ?"),
        ("研究判断", "SELECT MAX(created_at) d FROM research_note_versions WHERE symbol = ?"),
    ]
    for label, query in sources:
        latest = conn.execute(query, (symbol,)).fetchone()["d"]
        if latest and (since is None or latest > since):
            changes.append({"类型": label, "时间": latest})
    return changes


def record_research_update(conn: sqlite3.Connection, symbol: str, data_type: str,
                           status: str, detail: str | None = None) -> None:
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("""INSERT OR REPLACE INTO research_updates
        (symbol, data_type, checked_at, status, detail) VALUES (?, ?, ?, ?, ?)""",
        (symbol, data_type, checked_at, status, detail))
    conn.execute("""INSERT INTO research_update_runs
        (symbol, data_type, checked_at, status, detail) VALUES (?, ?, ?, ?, ?)""",
        (symbol, data_type, checked_at, status, detail))
    conn.commit()


def load_research_updates(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "data_type", "checked_at", "status", "detail"])
    placeholders = ",".join("?" for _ in symbols)
    return pd.read_sql_query(
        f"SELECT * FROM research_updates WHERE symbol IN ({placeholders})",
        conn, params=symbols)


def load_research_update_runs(conn: sqlite3.Connection, symbol: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM research_update_runs"
    params: list[str] = []
    if symbol:
        query += " WHERE symbol = ?"
        params.append(symbol)
    query += " ORDER BY id DESC"
    return pd.read_sql_query(query, conn, params=params)


def insert_financial_snapshot(conn: sqlite3.Connection, metadata: dict,
                              facts: list[dict]) -> int:
    """保存一版财务事实；snapshot_id 不复用，因此旧抓取版本始终保留。"""
    conn.execute(
        """INSERT INTO financial_statement_snapshots
           (snapshot_id, symbol, frequency, currency, trading_currency, source, source_url,
            published_at, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        tuple(metadata.get(k) for k in (
            "snapshot_id", "symbol", "frequency", "currency", "trading_currency", "source",
            "source_url", "published_at", "captured_at")),
    )
    conn.executemany(
        """INSERT INTO financial_facts
           (snapshot_id, period_start, period_end, statement, metric, value, unit, raw_label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(metadata["snapshot_id"], *(fact.get(k) for k in (
            "period_start", "period_end", "statement", "metric", "value", "unit",
            "raw_label"))) for fact in facts],
    )
    conn.commit()
    return len(facts)


def latest_statement_capture(conn: sqlite3.Connection, symbol: str,
                             frequency: str = "quarterly") -> str | None:
    row = conn.execute(
        """SELECT MAX(captured_at) AS d FROM financial_statement_snapshots
           WHERE symbol = ? AND frequency = ?""", (symbol, frequency)).fetchone()
    return row["d"]


def load_latest_financial_facts(conn: sqlite3.Connection, symbol: str,
                                frequency: str = "quarterly") -> pd.DataFrame:
    """读取单个最新快照的全部事实；不跨快照补空值。"""
    return pd.read_sql_query(
        """SELECT f.*, s.symbol, s.frequency, s.currency, s.trading_currency,
                  s.source, s.source_url,
                  s.published_at, s.captured_at
           FROM financial_facts AS f
           JOIN financial_statement_snapshots AS s USING (snapshot_id)
           WHERE s.snapshot_id = (
               SELECT snapshot_id FROM financial_statement_snapshots
               WHERE symbol = ? AND frequency = ?
               ORDER BY captured_at DESC, snapshot_id DESC LIMIT 1
           )
           ORDER BY f.period_end, f.statement, f.metric""",
        conn, params=[symbol, frequency])


def load_latest_quarterly_financials(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """把最新季度事实快照展开为 period_end × metric，元数据保留在 attrs。"""
    facts = load_latest_financial_facts(conn, symbol, "quarterly")
    if facts.empty:
        return pd.DataFrame()
    frame = facts.pivot_table(index="period_end", columns="metric", values="value",
                              aggfunc="first", dropna=False).sort_index()
    first = facts.iloc[0]
    frame.attrs.update({k: first[k] for k in (
        "snapshot_id", "symbol", "currency", "trading_currency", "source", "source_url",
        "published_at", "captured_at")})
    return frame


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


def mark_channel_delivered(conn: sqlite3.Connection, ids: list[int], channel: str) -> None:
    """记录某一渠道已成功送达；重复调用幂等。"""
    if not ids:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        """INSERT OR IGNORE INTO notification_deliveries
           (signal_id, channel, delivered_at) VALUES (?, ?, ?)""",
        [(signal_id, channel, now) for signal_id in ids],
    )
    conn.commit()


def delivered_signal_ids(conn: sqlite3.Connection, ids: list[int], channel: str) -> set[int]:
    if not ids:
        return set()
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT signal_id FROM notification_deliveries
            WHERE channel = ? AND signal_id IN ({placeholders})""",
        [channel, *ids],
    ).fetchall()
    return {int(row["signal_id"]) for row in rows}


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
            market_cap, book_value, beta, revenue_growth, earnings_growth, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
            metrics.get("revenue_growth"), metrics.get("earnings_growth"),
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


def load_latest_fundamentals(
    conn: sqlite3.Connection,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """按 symbol 返回完整的最新基本面快照行。

    不使用 ``groupby().last()``：pandas 会逐列跳过空值，把新快照的日期
    和旧快照的估值拼在一起。这里按主键日期选整行，缺失字段保持缺失。
    """
    params: list = []
    where = ""
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        where = f"WHERE symbol IN ({placeholders})"
        params.extend(symbols)
    query = f"""
        SELECT f.*
        FROM fundamentals AS f
        JOIN (
            SELECT symbol, MAX(date) AS date
            FROM fundamentals
            {where}
            GROUP BY symbol
        ) AS latest
          ON latest.symbol = f.symbol AND latest.date = f.date
        ORDER BY f.symbol
    """
    return pd.read_sql_query(query, conn, params=params)


def upsert_financials(conn: sqlite3.Connection, symbol: str,
                      rows: list[tuple]) -> int:
    """写入年度财报数据，金额币种与证券交易币种分别保存。

    rows = [(fiscal_date, revenue, gross_profit, operating_income, net_income,
             currency, trading_currency, source, source_url, captured_at), ...]。
    (symbol, fiscal_date) 已存在则覆盖。返回写入行数。"""
    conn.executemany(
        """INSERT OR REPLACE INTO financials
           (symbol, fiscal_date, revenue, gross_profit, operating_income, net_income,
            currency, trading_currency, source, source_url, captured_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [(symbol, *r) for r in rows],
    )
    conn.commit()
    return len(rows)


def load_financials(conn: sqlite3.Connection,
                    symbol: str | None = None) -> pd.DataFrame:
    """按 symbol, fiscal_date 升序返回年度财报。"""
    query = "SELECT * FROM financials WHERE 1=1"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY symbol, fiscal_date"
    return pd.read_sql_query(query, conn, params=params)


def latest_financial_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    """返回该标的最新一条财报的 captured_at，无记录返回 None。"""
    row = conn.execute(
        "SELECT MAX(captured_at) AS d FROM financials WHERE symbol = ?", (symbol,)
    ).fetchone()
    return row["d"]
