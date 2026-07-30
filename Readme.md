# 个人投资平台

美股日线个人投资平台：每天收盘后自动拉取行情、运行策略、把买卖信号推送到 Telegram / 邮件，并提供本地 Streamlit 面板做策略复盘、稳健性检验、组合诊断与基本面沉淀。**只出信号，不自动下单**，最终决策由人做。

## 功能

- **数据**：yfinance 拉取美股 ETF / 个股日线，SQLite 本地存储，增量更新；回测与动量一律用复权价（含分红，TLT 等品种必须），展示用原始收盘价
- **策略**（11 个，可插拔，`config.yaml` 开关与调参）：
  - `sma_cross`：双均线金叉/死叉（20/60）— 仅观察
  - `momentum`：行业 ETF 12-1 月度动量轮动（近 12 个月收益、跳过最近 1 个月，取前 3，月度调仓）
  - `rsi_reversal`：RSI(14) 超卖回升买入 / 超买回落卖出 — 仅观察
  - `smart_dca`：智能定投——每月定投，死叉暂停积攒、金叉恢复补投
  - `dual_momentum`：GEM 双动量——月度持最强风险资产，动量转负切避险；现金感知（避险也转负则持短债 BIL 吃无风险利率）
  - `vix_regime`：VIX 情绪提醒——恐慌/自满区、期限结构倒挂 — 仅观察
  - `stock_momentum`：个股横截面动量——按成交额月度重建流动性池（point-in-time）+ 12-1 动量 + 大盘 200 日均线过滤 — 仅观察（敏感性检验显示超额几乎全来自 NVDA，不作实盘仓位）
  - `low_vol`：低波动因子——行业 ETF + 债/金里波动最低的前 3（首个非动量分散因子）
  - `cross_asset_mom`：跨资产 12-1 月度动量 + 绝对动量开关（稳健档，9 类低相关资产）
  - `aggressive_mom`：进攻档——成长 ETF 集中 top1 12-1 动量 + 现金感知避险（本质集中成长押注）
  - `canary_mom`：哨兵动量（Keller DAA/HAA 架构）——把「何时防守」与「持什么」拆开，与持仓无关的哨兵资产用快信号（13612W）管风险开关、选择层用慢信号（12-1）
- **回测**：单标的、组合轮动（换仓资金不出场）、智能定投、VIX 四种模式；统一复权价 + 可配置单边成本（`backtest.cost_bps`）；风险对比表（收益/年化/回撤/波动/夏普/Calmar）与同口径**公平基准**逐列比较（板块等权 / 池子等权 / 等权全资产——而非 SPY/QQQ 这类事后赢家）
- **稳健性检验**（回测组合页）：调仓日 timing luck 散布 + 错峰 tranching + walk-forward 分段 + 池子等权，**输出期望区间不出通过/不通过**——回答「报出来的数字有没有虚高」，而非「有没有 alpha」
- **模型组合**（相关性页）：多策略相关矩阵 + 低相关配方推荐 + 等权组合 vs 单策略/SPY/QQQ；内含杠杆分析（把夏普换成收益，仅分析用不改实盘信号）
- **通知**：Telegram Bot + 邮件（SMTP），每条信号带人话理由；未配置的渠道自动跳过
- **面板**：Streamlit 十页 —— 市场概览、信号历史、K线与信号、动量排名、市场筛选、避险手册、策略评分、策略相关性、回测、策略说明
- **幂等**：重复运行不重复入库、不重复推送；`--date` 可补跑历史日期，`--backfill` 补全历史信号

## 安装

### macOS 一键部署（推荐）

```bash
bash scripts/setup_mac.sh          # 默认每天 14:00（美西盘后 1h，yfinance 已定稿）运行
bash scripts/setup_mac.sh 08:30    # 或指定其他时间
```

脚本会创建 `.venv` 虚拟环境、安装依赖、注册 launchd 定时任务（比 cron 好在：Mac 睡眠中错过的任务会在唤醒后补跑）。另附 `scripts/dashboard.command`（双击开面板）、`scripts/run_now.command`（双击手动跑一次）。取消任务的命令见脚本输出。

### 手动安装

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 与 SMTP 配置（都可选）
```

Telegram 配置：跟 [@BotFather](https://t.me/BotFather) 建 bot 拿 token；跟 [@userinfobot](https://t.me/userinfobot) 拿自己的 chat id。不配置也能用，信号会打印到终端。

邮件配置（Gmail 为例）：在 [Google 账号 → 安全性 → 应用专用密码](https://myaccount.google.com/apppasswords) 生成 16 位应用专用密码（不是登录密码），然后在 `.env` 里填：

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
python run_daily.py                     # 更新行情 → 跑策略 → 推送当日信号
python run_daily.py --date 2026-07-03   # 补跑某天的信号
python run_daily.py --no-fetch          # 跳过数据更新（离线调试）
python run_daily.py --no-notify         # 只入库不推送
python run_daily.py --full-refresh      # 全量重拉行情（复权价随分红回溯变化，建议每季度一次）
python run_daily.py --backfill          # 把各策略全量历史信号补入库（标记已通知不推送），初始化信号历史

streamlit run quant/web/app.py          # 打开复盘面板
python -m pytest tests/                 # 跑单元测试
```

## 配置

`config.yaml` 里改 watchlist（按组：大盘 / 行业 / 主题 / 资产类 / 现金 / 哨兵 / 防守 / 跨资产 / 进攻成长 等 15 组）、策略参数与作用组、通知开关、以及 `model_portfolio.strategies`（相关性页推荐配方的默认成分）。当前覆盖约 46 只 ETF / 个股候选，全部可随时增删。

## 目录

```
run_daily.py        每日主入口（launchd 调用）
config.yaml         watchlist / 策略 / 模型组合配置
CLAUDE.md           工程与方法论文档（含关键设计决策、诚实评估口径）
PLAN.md             设计文档
quant/config.py     配置加载
quant/data/         yfinance 拉取 + SQLite 存储
quant/strategies/   11 个策略（base + 各策略；REGISTRY 注册）
quant/backtest/     回测引擎（单标的 / 组合轮动 / 定投 / 波动率缩放）
quant/analysis/     纯计算：market / scoring / screening / correlation / robustness / drawdowns
quant/notify/       Telegram + 邮件推送
quant/web/app.py    Streamlit 十页面板
tests/              单元测试
scripts/            macOS 部署与快捷入口
```

## 方法论要点（详见 CLAUDE.md）

- **诚实评估**：回测数字偏乐观处（幸存者偏差、收盘价成交、区间运气、调仓日运气）主动标注，宁可低估不虚高
- **公平基准**：判断「选择有没有加信息」一律对比"与策略共享候选名单的等权持有"，不拿 SPY/QQQ/XLK 这类事后赢家当基准
- **两根正交的稳健性轴**：walk-forward（换时段还成立吗）+ 调仓日 timing luck（换调仓日还成立吗），月频策略两根都要过
- **等权是难打败的基线**：信号精加工、组合构建优化在我们这种小宇宙上都不敌朴素等权（1/N 结果）；edge 来自能 HOLD 什么（资产类别广度）+ 简单稳健的默认
- **策略有时效性 → 组合而非选优**：单押一个策略是 specification risk，同时持有几个低相关策略更稳

## 免责声明

本工具产生的信号仅供参考，不构成投资建议；请人工确认后再操作，风险自负。
