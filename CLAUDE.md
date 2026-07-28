# CLAUDE.md

个人投资平台（美股日线）。**只出信号不自动下单**，用户在 Mac 上以 launchd 每日运行，
Telegram + 邮件推送信号，Streamlit 面板复盘。与用户沟通用中文。

## 常用命令

```bash
python -m pytest tests/ -q                # 单测（全绿是提交底线）
python run_daily.py --no-fetch --no-notify  # 离线跑流水线（容器内用这个）
python run_daily.py --date 2026-07-03     # 补跑某日信号（幂等）
python run_daily.py --full-refresh        # 全量重拉行情（复权价拼接错位，季度一次）
python run_daily.py --backfill             # 补全量历史信号入库（标记已通知不推送）
streamlit run quant/web/app.py            # 面板（市场概览/信号历史/K线/动量排名/市场筛选/避险手册/策略评分/策略相关性/回测/策略说明）
```

## 架构速览

- `run_daily.py`：主入口。拉数据 → 各策略 generate → 信号入库（唯一约束幂等）→ dispatch 推送
- `quant/config.py`：config.yaml + .env；`update_symbols` = watchlist + 各策略 universe_file
- `quant/data/`：yfinance 增量拉取（首拉空表报错）、SQLite（prices/signals 两表）
- `quant/strategies/`：基类 `generate(prices: dict[symbol, df]) -> list[Signal]`，对**全量历史**出信号；
  每日运行筛当天，回测用完整序列。注册在 `__init__.py` 的 REGISTRY
- `quant/backtest/engine.py`：单标的、组合轮动（同日先卖后买、资金不出场）、智能定投三种模拟；
  `vol_scaled_equity` 纯函数（无杠杆波动率缩放，降回撤/尾部，实测不提升夏普；仅分析用不改实盘信号）
- `quant/analysis/`：robustness.py（**稳健性检验**：调仓日 timing luck 散布 + 错峰 tranching +
  分段 + 池子等权公平基准；**输出期望区间不出通过/不通过**，定位见该文件文档头）、
  market.py（52周区间位置/行业宽度/收益率利差，纯计算给市场概览页用）、
  scoring.py（signal_forward_returns 逐信号算 5/20/60 日前瞻收益，给策略评分页用）、
  correlation.py（策略相关性/组合诊断：各策略权益曲线转日收益率→Pearson相关矩阵→等权组合分散效果；
  `suggest_low_corr_set` 贪心挑低相关成分，供相关性页的「🧺 模型组合」用——对 specification risk
  与「策略有时效性」的实操回答：不是找永远有效的那个，而是同时持有几个决策方式不同、相互低相关的）、
  screening.py（市场筛选：个股/板块当前强弱快照；综合分=动量半[12-1动量/52周位置/距均线三维横截面]+价值半[forward盈利收益率+EV/EBITDA收益率双口径的行业内百分位，抗一次性收益畸变；金融EV/EBITDA失效则只用forward]，当前基本面快照非point-in-time）
- `quant/web/app.py`：十页面板（市场概览/信号历史/K线/动量排名/市场筛选/避险手册/策略评分/策略相关性/回测/策略说明）；
  避险手册页（`analysis/drawdowns.py`）：SPY 识别历史下跌段→每段测各避险资产总回报→崩盘类型自动判定
  （闪崩/通缩型-TLT有效/通胀型-TLT失效需商品黄金），含当前进行中回撤的实时"对号入座"；
  回测页按策略分单标的/组合/智能定投/VIX 四种渲染模式，组合模式含「🧭 稳健性检验」折叠区；
  策略相关性页含「🧺 模型组合」：自选/推荐几个低相关策略等权，对比单策略与 SPY/QQQ 长持 + 分段名次

十一个策略：sma_cross、momentum（行业 12-1 月度动量轮动）、rsi_reversal、smart_dca（定投+死叉暂停金叉补投）、
dual_momentum（GEM；现金感知+BIL 现金等价：避险 TLT 也负则持短债 BIL 吃无风险利率而非 0% 现金，
月首日口径 +419%/夏普0.85，但月首日是最优调仓日，去 timing luck 的错峰口径约 +347%/0.82）、
vix_regime（情绪提醒）、stock_momentum（个股 12-1 动量+流动性池）、
low_vol（行业 ETF + 债金低波动因子，首个非动量分散因子；扩宇宙含 TLT/GLD 后与 momentum 相关约 0.26）、
cross_asset_mom（跨资产 12-1 月度动量+绝对动量开关，稳健档；9 类低相关资产宇宙。
**2026-07-27 撤回"首个跑赢等权基准"的说法**——按夏普只是打平 0.80 vs 0.81，且 timing luck
跨度 569bp 全平台最脆弱、月首日恰是最优日，错峰口径 +162%/0.66/0.33 三项全输基准）、
aggressive_mom（进攻档：成长 ETF 集中 top1 12-1 动量+现金感知避险+BIL 现金等价；本质集中成长押注。
**timing luck 检验最稳的策略**：错峰口径 +717%/年化20.0%/回撤-33.1%/夏普0.81/Calmar0.60 与月首日口径
（+737%/-34.3%/0.81/0.59）几乎一致，优势不靠调仓日运气。**但公平基准对比 2026-07-27**：跑赢 QQQ 属实
但 QQQ 非公平基准；对★成长池等权（+606%/0.85/0.49）全历史赢收益/回撤/Calmar 却**输夏普**，
且 walk-forward 分裂——2015-2020 收益打平、夏普 Calmar 回撤三项都输，超额只在 2021-2026。
四个口径里策略夏普最低（0.81<0.85<0.87<0.89）= 集中押注的标准签名，**"排名加了信息"未被证明**）、
canary_mom（**哨兵动量**，2026-07-27 新增；2026-07-28 起进入推荐模型组合并开启推送。Keller DAA/HAA 架构：把「何时防守」
从「持什么」里拆出来——与持仓无关的哨兵 TIP 用**快信号 13612W** 只管风险开关，选择层仍用**慢信号 12-1**。
进攻宇宙与 cross_asset_mom 完全相同便于 A/B。错峰口径 +208%/年化10.2%/回撤-19.8%/夏普0.97/Calmar0.52，
vs ★进攻池等权 +171%/9.0%/-23.0%/0.81/0.39 → **walk-forward 两段都赢超额/夏普/回撤三项**，
是平台上**唯一**做到这点的策略。诚实边界见 canary.py：21 段风险关闭里 20 段只有 1 个月、
**没躲开 2020-03 新冠崩盘**、关掉自指过滤是 in-sample 选择、绝对收益仍远输 SPY）。

**现金等价 BIL（防御吃无风险利率）**：三个防御策略"持现金"的地方，top1 全进全出的 dual_momentum /
aggressive_mom 改持 BIL（1-3月短债 ETF，近零波动零久期，2007+ 全历史 = SGOV 的长历史版）吃短债利率
（2015-2021 利率≈0 时 BIL≈现金，2022 起 ~5% 每年多赚几个点）。cross_asset_mom（top3）**不接**：其空槽是
零散 1/3、2/3，等权引擎无法干净建模，且整仓换仓日会把钱集中到不足 3 个的正动量赢家上（+234% 有一部分
来自这个计划外集中，接 BIL 反把数字降到 +200%）。BIL 数据缺失时所有策略回落到 0% 现金。

## 关键设计决策（改动前务必理解）

1. **回测与动量计算一律用 adj_close 总回报口径**（`strategies.base.price_series`）——TLT 等
   收益大头在票息，close 口径会把结论算反。K线展示与 Signal.price 用原始 close。
2. **成本**：`backtest.cost_bps`（单边万5）所有回测与基准统一收取；指标以投入本金为分母
   （`equity_metrics(equity, initial)`），否则建仓成本被首日权益吞掉。
3. **基准可比性**：期初一次性投入的策略只对比长持；定投基准只出现在 smart_dca 模式
   （投入节奏一致才公平）。**每个选择型策略都必须对比"与它共享候选名单的等权持有"**
   ——这是判断"排名有没有加信息"的唯一可信对比（板块等权 / 池子等权 / 等权全资产 /
   成长池等权）。**别拿 QQQ、XLK 这类事后赢家当基准**：aggressive_mom 跑赢 QQQ 属实，
   但换成成长池等权后 walk-forward 前半段就不成立了（2026-07-27 实测）。
   **结论要同时看夏普和 Calmar**：集中型策略常见"赢 Calmar 输夏普"，这是集中度换来的
   回撤形状而非风险调整效率——只报有利的那个指标就是虚高。
4. **幸存者偏差**：`universe_sp500.yaml` 是今天的成分快照，绝对收益虚高；选股池按当时
   成交额逐月重建（point-in-time）缓解前视。回测页有剔除标的多选框做敏感性检验。
5. **幂等**：signals 表 (date,symbol,strategy,direction) 唯一；未配置通知渠道=打印即视为
   已送达；渠道失败才留待重试。**run_daily 默认只入库当天信号**（`s.date == as_of`），
   所以「信号历史」页只累积平台实际运行过且当天有信号的记录——初始化或找回历史用
   `--backfill`（全量历史信号入库并标记已通知，不倒灌推送）。
6. **信号 reason 必须是人话**（含数值与理由），推送和面板直接展示。
7. **watchlist 的 `macro` 组是纯展示**（大盘指数/美元/黄金/原油/比特币/十年期与三月期美债
   收益率），不喂给任何策略，只供市场概览页的瓷砖和情绪红绿灯用。
8. **策略评分 ≠ 回测**：评分页用 `signal_forward_returns` 只看单条信号发出后 N 日涨跌
   （不含仓位/成本），回测是机械执行整套策略的资金曲线模拟——两者故意不同，互为补充。
9. **稳健性检验回答的是「数字有没有虚高」，不是「有没有 alpha」**（2026-07-27 定调）。
   11 年月频、收益按 regime 高度自相关 → 独立观测只有 2-3 个，**任何检验都没有统计功效
   去证明 alpha**。所以三根检验轴都只用来校准期望值，**输出期望区间、不出通过/不通过**：
   - **walk-forward**：是不是靠单一 regime 撑起来的（切前后两半或 4 段）。
   - **调仓日 timing luck**（Newfound / Hoffstein-Faber-Braun）：是不是恰好挑中了最好的
     调仓日。所有月频策略默认锚在"每月首个交易日"，这是个**未经检验的隐含选择**。
     实现：`strategies.base.month_anchors(index, offset)` + 各策略的 `rebalance_offset`
     参数（默认 0 = 原行为），offset 取 0/5/10/15（≈4 个周度错峰 tranche）。
     "去掉运气"的公平估计 = 4 个 tranche 各 1/4 资金独立跑再求和（tranching）。
     实测年化跨度：cross_asset_mom 569bp（最脆弱，结论被推翻）、aggressive_mom 447bp
     （但错峰后指标不变=真稳）、dual_momentum 209bp、momentum 141bp、low_vol 133bp。
     **散布大不等于策略差**——看错峰组合是否仍站得住。
   - **池子等权**：是不是宇宙本身好、而非选择能力（见决策 #3）。
   **别把"某段跑输"当死刑判决**：价值因子连输 13 年、动量 2009 年崩过，按"每段都要赢"
   的门槛连有几十年文献支撑的因子都会被误杀。walk-forward 本身分不出"真效应坐冷板凳"
   与"假象恰好在某段好看"——分开它们靠**先验**（12-1 动量有跨市场独立验证 → 先验强；
   "top2 比 top3 好"是我们自己在数据里翻出来的 → 先验为零）。判断留给人，工具只给区间。
   工具：`quant/analysis/robustness.py`，面板在回测页的「🧭 稳健性检验」折叠区。
10. **策略有时效性 → 组合而非选优**（2026-07-27）。既然事前分不清哪个策略在当季、而且
    我们的数据也证不了谁有 alpha，单押一个策略就是纯 specification risk。实测（全历史）：
    低相关 4 件套（low_vol / aggressive_mom / cross_asset_mom / smart_dca）等权
    = **夏普 0.96 / Calmar 0.60 / 回撤 -20.6%，两项都是全平台第一**，高于每个单策略、
    也高于 SPY(0.81) 与 QQQ(0.89)；walk-forward 两段夏普第 2/7 与第 1/7，是**平台上唯一
    两段都稳居前列的东西**。但必须同时说清三条：(a) 收益 +280% 仍**输 SPY +336% / QQQ +621%**，
    这是风险调整的胜利不是增长的胜利；(b) 组合夏普高于每个成分**主要是分散的数学结果，
    不是 alpha**；(c) 成分沿用月首日调仓，继承了 dual_momentum / cross_asset_mom 的 timing luck，
    且低相关成分是拿全历史相关矩阵挑的（轻度 in-sample）。
    别因为某成分近期跑输就换掉它——按近期表现换策略本身就是过拟合。
    **canary_mom 加入后的复测（2026-07-28）**：它与 cross_asset_mom 相关 **0.76**（共享宇宙），
    所以是**替代**而非补位。三个值得记的配方（成分沿用月首日口径，仍带 timing luck）：
    - **canary 换掉 cross_asset**（low_vol/aggressive/canary/smart_dca）：夏普 1.01、Calmar 0.67、
      回撤 -18.0%，**且两个半段的夏普都是全场第一**（原 4 件套在 2015-2020 输给 QQQ）→ 严格优于原配方。
      **✅ 2026-07-28 已采纳为默认配方**，显式写在 `config.yaml` 的 `model_portfolio.strategies`
      （相关性页「🧺 模型组合」读它当默认勾选；`cfg.model_portfolio`）。配方是会随结论变的决定，
      所以像策略参数一样版本化、可追溯，而不是藏在面板代码里。canary_mom 同步开启 notify
      ——推荐一个策略却不推它的信号是自相矛盾的（单测 `test_config_model_portfolio_*` 守这条一致性）。
    - **canary + aggressive + low_vol**（平均相关 0.38，最低）：夏普 **1.03**、Calmar 0.73、
      回撤 -18.2%、+322% —— 全历史夏普最高的配方。
    - **canary + aggressive 双腿杠铃**：**+454% / 年化 16.0% / 回撤 -21.2% / 夏普 0.95 / Calmar 0.75**，
      **在每一项指标上都优于 SPY 长持**（+336%/13.6%/-33.7%/0.81/0.40）——这是平台上第一个做到
      "收益和回撤同时赢 SPY"的东西，正对[[stock-eval-benchmarks]]要的"风险可控的增长"。
      代价：只有 2 个成分 = specification risk 最高，且 aggressive_mom 自身的超额是 regime 依赖的。

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

- **stock_momentum 敏感性检验已结论：仅观察，不作实盘仓位。** 剔除 NVDA 单只标的即让
  2015–2020 收益跑输 SPY/QQQ，且剔除后连"池子等权"基准都跑不赢——超额全部来自 NVDA
  集中暴露，12-1 排名未加信息（等权反而更强）。不要凭其回测曲线给该策略分配真实资金。
- 未做/候选：期权链快照信号（covered call 权利金提醒，yfinance 可拉）、`next_open`
  次日开盘成交选项（低频策略影响小，用户已确认暂不需要）、点对点历史成分数据。
- **板块 momentum 已由 63日/每日 改为 252/skip21 月度**（旧版跑输板块等权，whipsaw +
  短期反转所致；改造后总收益 +264%、Calmar 0.38，交易 932→84 次）。timing luck 检验里
  它是唯一**被低估**的策略：月首日是四个调仓日里最差的（+262% vs 第11日 +319%），
  错峰口径 +290%/年化12.5%——文档数字偏保守而非偏乐观。
- **回测页有可选「波动率缩放」开关**（无杠杆减仓，降回撤/尾部，实测不提升夏普；仅分析用
  不改实盘信号）。用户可调目标波动率与回看窗口；cap=1.0 只减仓不加杠杆（实测加杠杆有害）。
