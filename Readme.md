# 个人量化信号平台

美股日线信号平台：每天收盘后自动拉取行情、运行策略、把买卖信号推送到 Telegram，并提供本地 Streamlit 面板复盘。**只出信号，不自动下单**，最终决策由人做。

## 功能

- **数据**：yfinance 拉取美股 ETF 日线，SQLite 本地存储，增量更新
- **策略**（可插拔，`config.yaml` 开关与调参）：
  - `sma_cross`：双均线金叉/死叉（默认 20/60）
  - `momentum`：行业 ETF 动量轮动（近 3 个月收益排名前 3）
  - `rsi_reversal`：RSI(14) 超卖回升买入 / 超买回落卖出
- **回测**：信号驱动的多头回测，输出总收益、年化、最大回撤、夏普、胜率
- **通知**：Telegram Bot 推送，每条信号带人话理由；未配置 token 时打印到终端
- **面板**：Streamlit 三页 —— 信号历史、K线与信号标注、回测报告
- **幂等**：重复运行不重复入库、不重复推送；`--date` 可补跑历史日期

## 安装

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（可选）
```

Telegram 配置：跟 [@BotFather](https://t.me/BotFather) 建 bot 拿 token；跟 [@userinfobot](https://t.me/userinfobot) 拿自己的 chat id。不配置也能用，信号会打印到终端。

## 使用

```bash
python run_daily.py                 # 更新行情 → 跑策略 → 推送当日信号
python run_daily.py --date 2026-07-03   # 补跑某天的信号
python run_daily.py --no-fetch     # 跳过数据更新（离线调试）
python run_daily.py --no-notify    # 只入库不推送

streamlit run quant/web/app.py     # 打开复盘面板
python -m pytest tests/            # 跑单元测试
```

### cron 定时（推荐）

美股收盘 16:00 ET ≈ 北京时间凌晨 4~5 点，建议北京时间每天早上 7:00 跑：

```cron
0 7 * * 2-6 cd /path/to/stock && python run_daily.py >> logs/cron.log 2>&1
```

（`2-6` = 周二至周六，对应美股周一至周五的收盘。）

## 配置

`config.yaml` 里改 watchlist（按组：大盘/行业/主题/资产类 ETF）、策略参数与作用组、通知开关。当前默认 21 只 ETF，全部可随时增删。

## 目录

```
run_daily.py        每日主入口（cron 调用）
config.yaml         watchlist 与策略配置
quant/data/         yfinance 拉取 + SQLite 存储
quant/strategies/   策略（base + 三个内置）
quant/backtest/     回测引擎
quant/notify/       Telegram 推送
quant/web/          Streamlit 面板
tests/              单元测试
PLAN.md             设计文档
```

## 免责声明

本工具产生的信号仅供参考，不构成投资建议；请人工确认后再操作，风险自负。
