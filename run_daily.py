#!/usr/bin/env python3
"""每日信号主入口（cron 调用）。

流程：增量更新行情 → 抓取基本面快照 → 跑启用的策略 → 新信号入库（幂等）→ Telegram 推送。
重复运行不会重复入库或重复推送；--date 可补跑历史某天的信号。
"""

import argparse
import logging
import sys
from datetime import date as _date
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
    parser.add_argument("--full-refresh", action="store_true",
                        help="全量重拉行情（修正复权价的增量拼接错位，建议每季度跑一次）")
    parser.add_argument("--backfill", action="store_true",
                        help="把各策略全量历史信号一次性补入库（标记为已通知，不推送），"
                             "用于初始化或找回信号历史")
    parser.add_argument("--no-fundamentals", action="store_true",
                        help="跳过基本面抓取")
    parser.add_argument("--fundamentals-only", action="store_true",
                        help="只抓基本面然后退出（用于单独补抓）")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config()
    conn = store.connect(cfg.db_path)

    failed: list[str] = []
    if not args.no_fetch:
        symbols = cfg.update_symbols
        log.info("%s %d 个标的行情…", "全量重拉" if args.full_refresh else "更新", len(symbols))
        total, failed = fetcher.update_all(conn, symbols, cfg.history_start,
                                           full=args.full_refresh)
        log.info("行情更新完成，共写入 %d 行，失败 %d 个", total, len(failed))

    # ── 基本面快照 ──
    run_fundamentals = (not args.no_fetch and not args.no_fundamentals) or args.fundamentals_only
    if run_fundamentals:
        try:
            fund_symbols = cfg.universe_symbols("universe_sp500.yaml")
            as_of_fund = args.date or _date.today().isoformat()
            log.info("抓取 %d 个个股基本面快照…", len(fund_symbols))
            fund_ok, fund_fail = fetcher.update_fundamentals(conn, fund_symbols, as_of_fund)
            log.info("基本面更新完成：成功 %d，失败 %d", fund_ok, len(fund_fail))
        except Exception:  # noqa: BLE001
            log.error("基本面抓取整体异常，不影响信号主流程", exc_info=True)
    if args.fundamentals_only:
        return 0

    prices = {s: store.load_prices(conn, s) for s in cfg.update_symbols}
    prices = {s: df for s, df in prices.items() if not df.empty}
    if not prices:
        log.error("库内没有任何行情数据，退出")
        return 1

    as_of = args.date or max(df.index.max() for df in prices.values()).strftime("%Y-%m-%d")
    log.info("信号日期: %s%s", as_of, "（backfill：补全量历史信号）" if args.backfill else "")

    all_new: list = []
    for name, params in cfg.enabled_strategies():
        strat = strategies.build(name, params)
        group_symbols = cfg.symbols_for(params.get("groups", []))
        if params.get("universe_file"):
            group_symbols += [s for s in cfg.universe_symbols(params["universe_file"])
                              if s not in group_symbols]
        group_prices = {s: prices[s] for s in group_symbols if s in prices}
        sigs = strat.generate(group_prices)
        if not args.backfill:
            sigs = [s for s in sigs if s.date == as_of]
        log.info("%s: %d 条%s信号", name, len(sigs), "历史" if args.backfill else "当日")
        all_new.extend(sigs)

    inserted = store.insert_signals(conn, all_new)
    log.info("新入库信号 %d 条（重复 %d 条已忽略）", inserted, len(all_new) - inserted)

    if args.backfill:
        marked = store.mark_all_notified(conn)
        log.info("backfill 完成：%d 条历史信号已标记为已通知（不推送）", marked)
        return 0

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
            shown = ", ".join(failed[:20]) + (f" 等 {len(failed)} 个" if len(failed) > 20 else "")
            dispatch(cfg, "⚠️ 量化数据更新失败", f"⚠️ 数据更新失败: {shown}，信号可能不完整")

    return 0


if __name__ == "__main__":
    sys.exit(main())
