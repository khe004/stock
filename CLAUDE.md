# CLAUDE.md

个人量化信号平台（美股日线）。**只出信号不自动下单**，用户在 Mac 上以 launchd 每日运行，
Telegram + 邮件推送信号，Streamlit 面板复盘。与用户沟通用中文。

## 常用命令

```bash
python -m pytest tests/ -q                # 单测（全绿是提交底线）
python run_daily.py --no-fetch --no-notify  # 离线跑流水线（容器内用这个）
python run_daily.py --date 2026-07-03     # 补跑某日信号（幂等）
python run_daily.py --full-refresh        # 全量重拉行情（复权价拼接错位，季度一次）
streamlit run quant/web/app.py            # 面板（市场概览/信号历史/K线/动量排名/策略评分/回测/策略说明）
```

## 架构速览

- `run_daily.py`：主入口。拉数据 → 各策略 generate → 信号入库（唯一约束幂等）→ dispatch 推送
- `quant/config.py`：config.yaml + .env；`update_symbols` = watchlist + 各策略 universe_file
- `quant/data/`：yfinance 增量拉取（首拉空表报错）、SQLite（prices/signals 两表）
- `quant/strategies/`：基类 `generate(prices: dict[symbol, df]) -> list[Signal]`，对**全量历史**出信号；
  每日运行筛当天，回测用完整序列。注册在 `__init__.py` 的 REGISTRY
- `quant/backtest/engine.py`：单标的、组合轮动（同日先卖后买、资金不出场）、智能定投三种模拟
- `quant/analysis/`：market.py（52周区间位置/行业宽度/收益率利差，纯计算给市场概览页用）、
  scoring.py（signal_forward_returns 逐信号算 5/20/60 日前瞻收益，给策略评分页用）
- `quant/web/app.py`：七页面板（市场概览/信号历史/K线/动量排名/策略评分/回测/策略说明）；
  回测页按策略分单标的/组合/智能定投/VIX 四种渲染模式

七个策略：sma_cross、momentum（行业轮动）、rsi_reversal、smart_dca（定投+死叉暂停金叉补投）、
dual_momentum（GEM）、vix_regime（情绪提醒）、stock_momentum（个股 12-1 动量+流动性池）。

## 关键设计决策（改动前务必理解）

1. **回测与动量计算一律用 adj_close 总回报口径**（`strategies.base.price_series`）——TLT 等
   收益大头在票息，close 口径会把结论算反。K线展示与 Signal.price 用原始 close。
2. **成本**：`backtest.cost_bps`（单边万5）所有回测与基准统一收取；指标以投入本金为分母
   （`equity_metrics(equity, initial)`），否则建仓成本被首日权益吞掉。
3. **基准可比性**：期初一次性投入的策略只对比长持；定投基准只出现在 smart_dca 模式
   （投入节奏一致才公平）。stock_momentum 另有"池子等权"基准——与策略共享候选名单的
   幸存者偏差，是判断"排名有没有加信息"的唯一可信对比。
4. **幸存者偏差**：`universe_sp500.yaml` 是今天的成分快照，绝对收益虚高；选股池按当时
   成交额逐月重建（point-in-time）缓解前视。回测页有剔除标的多选框做敏感性检验。
5. **幂等**：signals 表 (date,symbol,strategy,direction) 唯一；未配置通知渠道=打印即视为
   已送达；渠道失败才留待重试。
6. **信号 reason 必须是人话**（含数值与理由），推送和面板直接展示。
7. **watchlist 的 `macro` 组是纯展示**（大盘指数/美元/黄金/原油/比特币/十年期与三月期美债
   收益率），不喂给任何策略，只供市场概览页的瓷砖和情绪红绿灯用。
8. **策略评分 ≠ 回测**：评分页用 `signal_forward_returns` 只看单条信号发出后 N 日涨跌
   （不含仓位/成本），回测是机械执行整套策略的资金曲线模拟——两者故意不同，互为补充。

## 容器环境（Claude Code 云端）注意

- **代理不通 Yahoo/行情站点**，真实数据拉不了。验证用合成数据：scratchpad 有 seed 脚本
  灌 prices 表（21 只 ETF + ^VIX/^VIX3M + 60 只个股 + 10 个 macro 指数/资产），然后 `--no-fetch` 跑。
- 面板验证：`streamlit.testing.v1.AppTest` 逐页/逐策略跑；截图用 Playwright
  （executablePath=/opt/pw-browsers/chromium，selectbox 用 `[data-testid="stSelectbox"]`，
  暗色主题 `--theme.base dark` + `color_scheme='dark'`）。
- **别用 `pkill -f "streamlit"`**——模式会匹配到自己的 bash 把 shell 杀掉（exit 144）。
  换端口另起即可。
- YAML 名单里 `ON`/`NO` 类代码必须加引号（否则解析成布尔值）。
- **.gitignore 模式必须根锚定**（`/data/` 而非 `data/`）——曾因 `data/` 误匹配
  `quant/data/` 代码目录导致其从未被提交，容器一切正常而用户 clone 后缺文件。
  提交前用 `git ls-files --others --exclude-standard` 查漏网文件。
- 用户 Mac 是自带 bash 3.2：shell 脚本里变量一律 `${VAR}`，别让全角字符紧贴变量。

## 工作流约定

- 开发在 `claude/personal-quant-platform-plan-2xudfb` 分支，**每个功能完成后 commit →
  push → merge 进 master → push master**（用户已授权，master 是用户使用的分支）。
- 用户重视诚实评估：回测数字偏乐观的地方（幸存者偏差、收盘价成交、区间运气）要主动
  说明，宁可低估不可虚高。面板改动发截图给用户确认。
- 暗色/亮色主题都要可读：表格高亮用 rgba 半透明背景，前景色经 `st.context.theme` 切换。

## 当前状态与待定事项

- 用户正在真实数据上做 stock_momentum 的敏感性检验（剔除大赢家看超额是否塌掉），
  结论将决定该策略定位（卫星仓位 vs 仅观察）。
- 未做/候选：期权链快照信号（covered call 权利金提醒，yfinance 可拉）、`next_open`
  次日开盘成交选项（低频策略影响小，用户已确认暂不需要）、点对点历史成分数据。
