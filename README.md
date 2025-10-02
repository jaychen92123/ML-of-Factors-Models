# ML-of-Factors-Models

step1.爬蟲得到檔案"WebData.csv"

step2.讀取"WebData.csv"後計算因子並匯出檔案'withfactor.csv'

step3.讀取檔案'withfactor.csv'並使用LSTM和CNN訓練並回測績效

# 總結：

## 因子訊號

首先說明一下我的因子是["open-avg_vwap","cs_rank_open-avg_vwap*close-vwap","corr_hv_5","rank_std_high_5"]

VWAP：跟一般均價不同，會讓成交量大的那個價格權重更多

 $\mathrm{VWAP}=\frac{\sum{t=1}^{T} P_t\,Q_t}{\sum_{t=1}^{T} Q_t}$

open - avg_vwap：開盤減去五日平均vwap，看開盤離這條均線的距離

cs_rank_open-avg_vwap*close-vwap：是橫截面排序open-avg_vwap*close-vwap

corr_hv_5：是五日來最高價與成交量的相關係數

rank_std_high_5：是個股序列排序，看今天的五日最高價標準差是這五天的第幾名的概念

總的來說，我預期這些因子能夠掌握過熱股或過度偏離大部分人的持有成本價格，另一方向同理
