# ML-of-Factors-Models

step1.爬蟲得到檔案"WebData.csv"

step2.讀取"WebData.csv"後計算因子並匯出檔案'withfactor.csv'

step3.讀取檔案'withfactor.csv'並使用LSTM和CNN訓練並回測績效

## 總結：

首先說明一下我的因子是["open-avg_vwap","cs_rank_open-avg_vwap*close-vwap","corr_hv_5","rank_std_high_5"]

VWAP：跟一般均價不同，會讓成交量大的那個價格權重更多

 $\mathrm{VWAP}=\frac{\sum{t=1}^{T} P_t\,Q_t}{\sum_{t=1}^{T} Q_t}$

open - avg_vwap：開盤減去五日平均vwap，看開盤離這條均線的距離

cs_rank_open-avg_vwap*close-vwap：是橫截面排序open-avg_vwap*close-vwap

corr_hv_5：是五日來最高價與成交量的相關係數

rank_std_high_5：是個股序列排序，看今天的五日最高價標準差是這五天的第幾名的概念

總的來說，我預期這些因子能夠掌握過熱股或過度偏離大部分人的持有成本價格，另一方向同理

訊號定義

VWAP（成交量加權均價）：放大高量價位的影響。

open−avg_vwap：今日開盤 − 近5日平均 VWAP；衡量開盤相對大多數持有成本的偏離幅度。

cs_rank_open-avg_vwap*close-vwap：在橫截面上，對 （open−avg_vwap）×（close−vwap） 排名；找出同日最偏離且伴隨收盤偏移的股票。

corr_hv_5：5 日內「最高價 vs 量」相關係數；量價同漲的過熱程度。

rank_std_high_5：以時間序列視角，對「5 日最高價標準差」做序列排名；波動/擴散是否異常。

訊號直觀

偏離越大（相對 VWAP），代表與主流成本差距大；易出現均值回歸或動能延續（視量價結構）。

量價正相關、短期高點波動放大 → 過熱傾向，隔日/數日回吐機率上升。

交易規則（範例）

當日收盤後計算 4 訊號 → 標準化/去極值 → 合成分數 score（線性加權或模型輸出）。

依 score 排名：做多 Top-N、做空 Bottom-N（等權或依信號強度配重）。

週期：日調或週調（降頻可降成本）；設不交易帶（|score|小於閾值不換倉）。

成本：交易成本/滑價以 bps 計入；單檔/產業/因子曝險做上限控制。

風控與組合

單檔權重上限、淨暴露/β 中性（選擇性）、產業中性（選擇性）。

風險目標：日波動、最大回撤、換手率門檻。

訊號衝突時以 turnover 成本與邊際貢獻排序決策。

回測與評估

資料分割：滾動/走勢外推（train/valid/test，避免洩漏）。

目標：例如未來 3 日報酬（或你現用的 horizon）。

指標：IC / RankIC、Precision@N、命中率、年化報酬/波動/Sharpe、最大回撤、Turnover、成本占比。

穩健性：視窗參數、N 值、成本假設做敏感度分析；檢查牛熊市分段績效。

可能風險

偏離訊號在漲停/跌停、流動性驟降時失靈。

量價過熱度在事件驅動（財報/公告）下可能持續而非回歸。

訊號共線性（VWAP 相關特徵彼此重疊）→ 建議做正則化或降維。

實作要點

avg_vwap：近5日 VWAP 的均值；vwap_t = 金額/成交量。

corr_hv_5：rolling corr(high, volume, 5)。

rank_std_high_5：rolling std(high, 5) 後做序列排名（窗口自訂）。

合成分數：

簡單法：score = w1*z1 + w2*z2 + w3*z3 + w4*z4（z 為標準化後訊號）。

進階法：用 XGBoost / LSTM 以 Sharpe-loss 訓練，score 直接當預期邊際報酬。

後續優化

週調倉 + 不交易帶降成本；

加入流動性/風險調整（如對 score 做波動或市值中性化）；

事件濾網（財報、除權息、處置股等）；

分 bucket 檢驗：依 open−avg_vwap 量化偏離分組觀察 PnL 單調性。

核心一句話：用「相對 VWAP 的偏離」搭配「量價過熱/波動異常」來挑出偏離過大的股票，做多/做空兩端，並透過週期與不交易帶控成本，以 IC/Sharpe 驗證其在不同市況下的穩健性。
