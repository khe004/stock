#!/usr/bin/env python3
"""每日信号主入口（cron 调用）。

流程：增量更新行情 → 跑启用的策略 → 新信号入库（幂等）→ Telegram 推送。
重复运行不会重复入库或重复推送；--date 可补跑历史某天的信号。
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant import strategies
from quant.config import ROOT, load_config
from quant.data import fetcher, store
from quant.notify import email, telegram
from quant.strategies.base import BUY

log = logging.getLogger("run_daily")


def setup_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "quant.log", encoding="utf-8"),
        ],
    )


def dispatch(cfg, subject: str, text: str) -> bool:
    """把消息发到所有启用渠道，全部送达才返回 True。
    某渠道失败时信号保持未通知，下次运行整体重发（成功过的渠道会收到重复）。"""
    ok = True
    if cfg.telegram_enabled:
        ok = telegram.send_message(text) and ok
    if cfg.email_enabled:
        ok = email.send_email(subject, text) and ok
    return ok


def format_message(rows) -> str:
    date = rows[0]["date"]
    lines = [f"📈 量化信号 {date}" if all(r["date"] == date for r in rows) else "📈 量化信号（含补跑）"]
    for r in rows:
        icon = "🟢 买入" if r["direction"] == BUY else "🔴 卖出"
        lines.append(f"{icon} {r['symbol']} ${r['price']:.2f} [{r['strategy']}]")
        lines.append(f"    {r['reason']}")
    lines.append("\n⚠️ 信号仅供参考，请人工确认后操作")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="每日量化信号")
    parser.add_argument("--date", help="只保留该日期(YYYY-MM-DD)的信号，默认取行情最新一天（用于补跑）")
    parser.add_argument("--no-fetch", action="store_true", help="跳过数据更新，直接用库内数据")
    parser.add_argument("--no-notify", action="store_true", help="不推送，只入库")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config()
    conn = store.connect(cfg.db_path)

    failed: list[str] = []
    if not args.no_fetch:
        log.info("更新 %d 个标的行情…", len(cfg.all_symbols))
        total, failed = fetcher.update_all(conn, cfg.all_symbols, cfg.history_start)
        log.info("行情更新完成，共写入 %d 行，失败 %d 个", total, len(failed))

    prices = {s: store.load_prices(conn, s) for s in cfg.all_symbols}
    prices = {s: df for s, df in prices.items() if not df.empty}
    if not prices:
        log.error("库内没有任何行情数据，退出")
        return 1

    as_of = args.date or max(df.index.max() for df in prices.values()).strftime("%Y-%m-%d")
    log.info("信号日期: %s", as_of)

    all_new: list = []
    for name, params in cfg.enabled_strategies():
        strat = strategies.build(name, params)
        group_symbols = cfg.symbols_for(params.get("groups", []))
        group_prices = {s: prices[s] for s in group_symbols if s in prices}
        sigs = [s for s in strat.generate(group_prices) if s.date == as_of]
        log.info("%s: %d 条当日信号", name, len(sigs))
        all_new.extend(sigs)

    inserted = store.insert_signals(conn, all_new)
    log.info("新入库信号 %d 条（重复 %d 条已忽略）", inserted, len(all_new) - inserted)

    if not args.no_notify:
        pending = store.unnotified_signals(conn)
        if pending:
            subject = f"📈 量化信号 {as_of}（{len(pending)} 条）"
            ok = dispatch(cfg, subject, format_message(pending))
            if ok:
                store.mark_notified(conn, [r["id"] for r in pending])
            log.info("推送 %d 条信号%s", len(pending), "" if ok else "（部分渠道失败，下次运行重试）")
        else:
            log.info("今日无新信号")
        if failed:
            dispatch(cfg, "⚠️ 量化数据更新失败", f"⚠️ 数据更新失败: {', '.join(failed)}，信号可能不完整")

    return 0


if __name__ == "__main__":
    sys.exit(main())
