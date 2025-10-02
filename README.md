# ML-of-Factors-Models

step1.爬蟲得到檔案"WebData.csv"

step2.讀取"WebData.csv"後計算因子並匯出檔案'withfactor.csv'

step3.讀取檔案'withfactor.csv'並使用LSTM和CNN訓練並回測績效

# 總結：

透過CNN模型的結果普遍優於時序控制更強烈的LSTM

整體噪音還是過於強烈 可以加入更多中立與降頻處理

## 因子訊號

首先說明一下我的因子是["open-avg_vwap","cs_rank_open-avg_vwap*close-vwap","corr_hv_5","rank_std_high_5"]

VWAP：跟一般均價不同，會讓成交量大的那個價格權重更多

 $\mathrm{VWAP}=\frac{\sum{t=1}^{T} P_t\,Q_t}{\sum_{t=1}^{T} Q_t}$

open - avg_vwap：開盤減去五日平均vwap，看開盤離這條均線的距離

cs_rank_open-avg_vwap*close-vwap：是橫截面排序open-avg_vwap*close-vwap

corr_hv_5：是五日來最高價與成交量的相關係數

rank_std_high_5：是個股序列排序，看今天的五日最高價標準差是這五天的第幾名的概念

總的來說，我預期這些因子能夠掌握過熱股或過度偏離大部分人的持有成本價格，另一方向同理

## 交易規則

當日收盤後計算訊號 → 標準化/rank去規模 → 合成分數 score（模型輸出）

依 score 排名：做多 Top-N、做空 Bottom-N（等權配重）。

成本：交易成本/滑價以 bps 計入。

## 可能風險

偏離訊號在漲停/跌停、流動性驟降時失靈。

量價過熱度在事件驅動（財報/公告）下可能持續而非回歸。

## 結果與可改進部分

看起來因子在空方表現較優異，在多方較沒能得到優異的回報率，可能是一組空方偏置因子。

敏感度分析、檢查牛熊市分段績效，不過資料期間只有三年所以沒有做這一part

單檔權重上限、產業中性，增加風險控制與中立性

週期降頻、設不交易帶



