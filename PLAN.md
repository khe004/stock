# 个人量化信号平台 — 规划文档

## 目标

一个跑在自己电脑上的美股日线信号平台：每天收盘后自动拉数据、跑策略、把买卖信号推送到 Telegram，并提供一个本地 Web 面板用于查看信号历史、K 线和回测结果。**只给信号，不自动下单**，最终决策由人做。

## 核心决策（已确认）

| 决策项 | 选择 | 影响 |
|---|---|---|
| 市场 | 美股 | 数据源用 yfinance（免费日线） |
| 频率 | 日线，收盘后出信号 | 每天跑一次，无需实时数据 |
| 通知 | Telegram Bot + 本地 Web 面板 | 推送即时信号，面板用于复盘 |
| 部署 | 自己电脑，cron 定时 | 零成本；需注意开机时间 |

## 整体架构

```
┌─────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐
│ 数据获取     │──▶│ 本地存储  │──▶│ 策略引擎      │──▶│ 信号生成      │
│ (yfinance)  │   │ (SQLite) │   │ (可插拔策略)  │   │ (落库+去重)   │
└─────────────┘   └────┬─────┘   └──────┬───────┘   └──────┬───────┘
                       │                │                   │
                       │         ┌──────▼───────┐   ┌───────▼──────┐
                       │         │ 回测引擎      │   │ 通知推送      │
                       │         │ (策略验证)    │   │ (Telegram)   │
                       │         └──────────────┘   └──────────────┘
                       │
                ┌──────▼───────────────────────────┐
                │ Web 面板 (Streamlit)              │
                │ 信号列表 / K线图 / 回测报告 / 命中率 │
                └──────────────────────────────────┘
```

## 目录结构

```
stock/
├── config.yaml            # 关注列表(watchlist)、策略开关与参数、通知配置
├── run_daily.py           # 主入口：更新数据 → 跑策略 → 推送信号（cron 调用）
├── quant/
│   ├── data/
│   │   ├── fetcher.py     # yfinance 拉取日线 OHLCV，增量更新
│   │   └── store.py       # SQLite 读写（行情表、信号表）
│   ├── strategies/
│   │   ├── base.py        # Strategy 基类：generate(df) -> list[Signal]
│   │   ├── sma_cross.py   # 双均线金叉/死叉
│   │   ├── momentum.py    # 动量轮动（watchlist 内排名）
│   │   └── rsi_reversal.py# RSI 超卖反弹
│   ├── backtest/
│   │   └── engine.py      # 简单向量化回测：收益、最大回撤、胜率、夏普
│   ├── notify/
│   │   └── telegram.py    # Bot API 推送（requests 直调，无重依赖）
│   └── web/
│       └── app.py         # Streamlit 面板
├── tests/                 # 策略与回测的单元测试
└── legacy/                # 现有的 stock.py / candlestick.py 归档
```

## 技术选型

- **语言/环境**：Python 3.11+，`requirements.txt` 或 `uv` 管理依赖
- **数据源**：`yfinance`（免费、日线充足）。非官方接口偶有限流 → 本地缓存 + 增量更新 + 失败重试
- **存储**：SQLite 单文件。个人规模（几十只股票 × 十年日线）绰绰有余，零运维
- **回测**：先自研轻量向量化回测（pandas 实现，~200 行），指标算清楚即可；将来策略复杂了再考虑 vectorbt
- **面板**：Streamlit —— 几十行代码就有交互图表，适合个人使用，`streamlit run` 想看时再开
- **推送**：Telegram Bot API，用 `requests` 直接 POST，token 放环境变量/`.env`，不进仓库
- **调度**：cron。美股收盘 16:00 ET ≈ 北京时间凌晨 4~5 点，建议定在**北京时间早上（如 7:00）**跑，错过开机可手动补跑（`run_daily.py` 设计成幂等，重复跑不重复推送）

## 数据模型

**行情表 `prices`**：`symbol, date, open, high, low, close, adj_close, volume`（主键 symbol+date）

**信号表 `signals`**：
`id, date, symbol, strategy, direction(buy/sell), price, strength(0-1), reason(人话解释), notified_at`

信号带 `reason` 字段很重要——推送里写"AAPL：20日均线上穿60日均线，收盘 $XXX"，而不是只给一个代码，方便人工审核。

## 每日运行流程（run_daily.py）

1. 读 `config.yaml` 的 watchlist
2. 增量拉取各标的最新日线，写入 SQLite
3. 依次运行启用的策略，收集信号
4. 信号去重（同一天同标的同策略只发一次），写入信号表
5. 有新信号 → 汇总成一条 Telegram 消息推送；无信号 → 静默（可配置每日心跳）
6. 全程写日志，异常也推送 Telegram 提醒（比如数据拉取失败）

## 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M1 数据层 | fetcher + store + config + run_daily 骨架 | 一条命令拉全 watchlist 日线并入库 |
| M2 策略框架 | Strategy 基类 + 双均线策略 + 信号落库 | 跑一次能在库里看到历史金叉信号 |
| M3 回测 | 回测引擎 + 指标报告 | 双均线在 AAPL 十年数据上出收益/回撤/胜率 |
| M4 推送 | Telegram 通知 + 幂等去重 | 收盘后手机收到当日信号 |
| M5 面板 | Streamlit：信号列表、K线+信号标注、回测图 | 本地打开网页可复盘 |
| M6 复盘 | 信号命中率统计（N 日后涨跌验证） | 面板显示每个策略的历史命中率 |

M1–M4 完成即形成闭环（数据→信号→手机），M5/M6 是体验增强。

## 风险与注意事项

- **yfinance 限流/改版**：数据层做成接口抽象，将来可换 Alpha Vantage、Tiingo 等
- **过拟合**：回测好看 ≠ 未来赚钱。参数不要调到极致，优先简单、逻辑说得通的策略
- **电脑不开机**：cron 错过就没跑；`run_daily.py` 幂等 + 支持补跑历史日期即可兜底
- **信号仅供参考**：平台定位是提醒和纪律工具，不构成投资建议，最终下单永远人工确认

## 遗留代码处理

现有 `stock.py` / `candlestick.py` 依赖已失效的 `pandas_datareader` yahoo 接口和被移除的 `matplotlib.finance`，无法运行，移入 `legacy/` 归档，不再维护。
