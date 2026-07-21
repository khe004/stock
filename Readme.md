# 个人投资平台

美股日线个人投资平台：每天收盘后自动拉取行情、运行策略、把买卖信号推送到 Telegram，并提供本地 Streamlit 面板做策略复盘、组合诊断与基本面沉淀。**只出信号，不自动下单**，最终决策由人做。

## 功能

- **数据**：yfinance 拉取美股 ETF 日线，SQLite 本地存储，增量更新
- **策略**（可插拔，`config.yaml` 开关与调参）：
  - `sma_cross`：双均线金叉/死叉（默认 20/60）
  - `momentum`：行业 ETF 动量轮动（近 3 个月收益排名前 3）
  - `rsi_reversal`：RSI(14) 超卖回升买入 / 超买回落卖出
  - `smart_dca`：智能定投——每月定投提醒，死叉暂停积攒、金叉恢复补投
  - `dual_momentum`：GEM 双动量——月度持有最强风险资产，动量转负切换避险资产
  - `vix_regime`：VIX 情绪提醒——恐慌区进入/消退、自满区、期限结构倒挂/解除
  - `stock_momentum`：个股横截面动量——每月按成交额重建流动性池（point-in-time），
    池内 12-1 动量选前 6 只（单行业上限），大盘跌破 200 日均线整体切避险；
    候选超集见 `universe_sp500.yaml`（注意其幸存者偏差，评估以"池子等权"基准为准）
- **回测**：单标的、组合轮动（换仓资金不出场）、智能定投三种模式；统一用复权价
  （含分红，TLT 等品种必须）与可配置单边成本（`backtest.cost_bps`）；风险对比表
  （收益/年化/回撤/波动/夏普/Calmar）与同口径基准逐列比较
- **通知**：Telegram Bot 推送 + 邮件（SMTP），每条信号带人话理由；未配置的渠道自动跳过
- **面板**：Streamlit 七页 —— 市场概览（指数/资产瓷砖 + 情绪红绿灯）、信号历史、
  K线与信号标注、动量排名、策略评分（信号发出后 5/20/60 日表现）、回测报告、策略说明
- **幂等**：重复运行不重复入库、不重复推送；`--date` 可补跑历史日期

## 安装

### macOS 一键部署（推荐）

```bash
bash scripts/setup_mac.sh          # 默认每天 07:00 自动运行
bash scripts/setup_mac.sh 08:30    # 或指定其他时间
```

脚本会创建 `.venv` 虚拟环境、安装依赖、注册 launchd 定时任务（比 cron 好在：Mac 睡眠中错过的任务会在唤醒后补跑）。取消任务的命令见脚本输出。

### 手动安装

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（可选）
```

Telegram 配置：跟 [@BotFather](https://t.me/BotFather) 建 bot 拿 token；跟 [@userinfobot](https://t.me/userinfobot) 拿自己的 chat id。不配置也能用，信号会打印到终端。

邮件配置（Gmail 为例）：在 [Google 账号 → 安全性 → 两步验证 → 应用专用密码](https://myaccount.google.com/apppasswords) 生成一个 16 位应用专用密码（不是登录密码），然后在 `.env` 里填：

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=你的gmail地址
SMTP_PASSWORD=16位应用专用密码
EMAIL_TO=收件地址（多个用英文逗号分隔）
```

Telegram 和邮件各自可在 `config.yaml` 的 `notify:` 下开关；某渠道发送失败时信号保持未通知状态，下次运行自动重试。

## 使用

```bash
python run_daily.py                 # 更新行情 → 跑策略 → 推送当日信号
python run_daily.py --date 2026-07-03   # 补跑某天的信号
python run_daily.py --no-fetch     # 跳过数据更新（离线调试）
python run_daily.py --no-notify    # 只入库不推送
python run_daily.py --full-refresh # 全量重拉行情（复权价随分红回溯变化，建议每季度跑一次）
python run_daily.py --backfill      # 把各策略全量历史信号补入库（标记已通知不推送），初始化信号历史

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
