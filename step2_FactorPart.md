```python
import pandas as pd
import numpy as np
```

```python
df = pd.read_csv("WebData.csv")
```

```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>日期</th>
      <th>代號</th>
      <th>名稱</th>
      <th>成交股數</th>
      <th>成交筆數</th>
      <th>成交金額</th>
      <th>開盤價</th>
      <th>最高價</th>
      <th>最低價</th>
      <th>收盤價</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-02</td>
      <td>1101</td>
      <td>台泥</td>
      <td>18470566</td>
      <td>6251</td>
      <td>813465904</td>
      <td>43.80</td>
      <td>44.15</td>
      <td>43.80</td>
      <td>44.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>1102</td>
      <td>亞泥</td>
      <td>8890485</td>
      <td>4391</td>
      <td>433140140</td>
      <td>48.10</td>
      <td>49.00</td>
      <td>48.05</td>
      <td>48.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-02</td>
      <td>1103</td>
      <td>嘉泥</td>
      <td>2194046</td>
      <td>883</td>
      <td>49255964</td>
      <td>22.40</td>
      <td>22.70</td>
      <td>22.35</td>
      <td>22.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-02</td>
      <td>1104</td>
      <td>環泥</td>
      <td>867516</td>
      <td>384</td>
      <td>17026458</td>
      <td>19.60</td>
      <td>19.70</td>
      <td>19.55</td>
      <td>19.65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-02</td>
      <td>1108</td>
      <td>幸福</td>
      <td>310216</td>
      <td>162</td>
      <td>2593989</td>
      <td>8.38</td>
      <td>8.45</td>
      <td>8.28</td>
      <td>8.37</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1183580</th>
      <td>2024-12-31</td>
      <td>9944</td>
      <td>新麗</td>
      <td>115881</td>
      <td>134</td>
      <td>2329626</td>
      <td>20.05</td>
      <td>20.30</td>
      <td>19.95</td>
      <td>20.30</td>
    </tr>
    <tr>
      <th>1183581</th>
      <td>2024-12-31</td>
      <td>9945</td>
      <td>潤泰新</td>
      <td>4357585</td>
      <td>3703</td>
      <td>186985362</td>
      <td>43.30</td>
      <td>43.35</td>
      <td>42.65</td>
      <td>42.90</td>
    </tr>
    <tr>
      <th>1183582</th>
      <td>2024-12-31</td>
      <td>9946</td>
      <td>三發地產</td>
      <td>156969</td>
      <td>161</td>
      <td>3551688</td>
      <td>22.95</td>
      <td>22.95</td>
      <td>22.50</td>
      <td>22.60</td>
    </tr>
    <tr>
      <th>1183583</th>
      <td>2024-12-31</td>
      <td>9955</td>
      <td>佳龍</td>
      <td>123625</td>
      <td>178</td>
      <td>3521198</td>
      <td>28.70</td>
      <td>28.70</td>
      <td>28.30</td>
      <td>28.50</td>
    </tr>
    <tr>
      <th>1183584</th>
      <td>2024-12-31</td>
      <td>9958</td>
      <td>世紀鋼</td>
      <td>1775254</td>
      <td>1727</td>
      <td>289819602</td>
      <td>164.00</td>
      <td>165.00</td>
      <td>162.00</td>
      <td>164.00</td>
    </tr>
  </tbody>
</table>
<p>1183585 rows × 10 columns</p>
</div>



```python
def add_factor(df , win=5):
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    
    df['vwap'] = df.iloc[:]["成交金額"]/df.iloc[:]["成交股數"]
    df["sum_value_5d"] = (
    df.groupby("代號")["成交金額"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())
    )
    df["sum_vol_5d"] = (
        df.groupby("代號")["成交股數"]
          .transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())
    )
    df["vwap_5d"] = df["sum_value_5d"] / df["sum_vol_5d"]
    
    df["open-avg_vwap"] = df["開盤價"] - df["vwap_5d"]
    
    df["close-vwap"] = df["收盤價"] - df["vwap"]
    
    grp = df.groupby("日期")["open-avg_vwap"]
    r = grp.rank(method="average")
    n = grp.transform("size")
    df["cs_rank_open-avg_vwap"] = (r - 1) / (n - 1)
    
    grp = df.groupby("日期")["close-vwap"]
    r = grp.rank(method="average")          
    n = grp.transform("size")
    df["cs_rank_close-vwap"] = (r - 1) / (n - 1)
    
    df["cs_rank_open-avg_vwap*close-vwap"] = ((df["cs_rank_open-avg_vwap"] * df["cs_rank_close-vwap"])-0.5) * -2
    
    df = df.sort_values(['代號','日期'])

    df['std_high_5'] = (
        df.groupby('代號')['最高價']
          .transform(lambda s: s.shift(1).rolling(win, min_periods=win).std())
    )

    df['rank_std_high_5'] = (
        df.groupby('日期')['std_high_5'].rank(pct=True)
    )

    df['corr_hv_5'] = (
        df.groupby('代號', group_keys=False)
          .apply(lambda g: g['最高價'].rolling(win, min_periods=win).corr(g['成交股數']/1000))
          .reset_index(level=0, drop=True).replace([np.inf, -np.inf], 0)
    )
    
    grp = df.groupby("代號", group_keys=False)
    open1 = grp["開盤價"].shift(-1)
    close1 = grp["收盤價"].shift(-1)
    close3 = grp["收盤價"].shift(-3)

    df["ret1d"] = (close1 / open1) - 1
    df["ret3d"] = (close3 / open1) - 1
    
    df = df.drop(columns=["最低價","最高價","成交金額","成交股數","成交筆數","sum_value_5d","sum_vol_5d","vwap","vwap_5d","std_high_5","cs_rank_open-avg_vwap","cs_rank_close-vwap"])
    df = df.dropna()
    
    df = df.sort_values(["日期","代號"], ignore_index=True)
    
    return df
```

```python
df1 = add_factor(df)
```

```python
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>日期</th>
      <th>代號</th>
      <th>名稱</th>
      <th>開盤價</th>
      <th>收盤價</th>
      <th>open-avg_vwap</th>
      <th>close-vwap</th>
      <th>cs_rank_open-avg_vwap*close-vwap</th>
      <th>rank_std_high_5</th>
      <th>corr_hv_5</th>
      <th>ret1d</th>
      <th>ret3d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-09</td>
      <td>1101</td>
      <td>台泥</td>
      <td>43.40</td>
      <td>43.45</td>
      <td>-0.281959</td>
      <td>0.034551</td>
      <td>0.426299</td>
      <td>0.534685</td>
      <td>0.068691</td>
      <td>0.001151</td>
      <td>0.025316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-09</td>
      <td>1102</td>
      <td>亞泥</td>
      <td>48.55</td>
      <td>48.95</td>
      <td>0.125706</td>
      <td>0.170082</td>
      <td>-0.522899</td>
      <td>0.690502</td>
      <td>0.784326</td>
      <td>-0.007150</td>
      <td>0.001021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-09</td>
      <td>1103</td>
      <td>嘉泥</td>
      <td>21.40</td>
      <td>21.55</td>
      <td>-0.455489</td>
      <td>0.139096</td>
      <td>0.473181</td>
      <td>0.589114</td>
      <td>0.916373</td>
      <td>-0.002320</td>
      <td>0.004640</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-09</td>
      <td>1104</td>
      <td>環泥</td>
      <td>19.05</td>
      <td>19.05</td>
      <td>-0.378377</td>
      <td>-0.004739</td>
      <td>0.648784</td>
      <td>0.397012</td>
      <td>0.968930</td>
      <td>-0.002611</td>
      <td>-0.002611</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-09</td>
      <td>1108</td>
      <td>幸福</td>
      <td>8.15</td>
      <td>8.18</td>
      <td>-0.157160</td>
      <td>0.007959</td>
      <td>0.350237</td>
      <td>0.115261</td>
      <td>0.668492</td>
      <td>-0.013317</td>
      <td>-0.004843</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1146045</th>
      <td>2024-12-26</td>
      <td>9944</td>
      <td>新麗</td>
      <td>20.10</td>
      <td>20.35</td>
      <td>0.082803</td>
      <td>0.014048</td>
      <td>0.596240</td>
      <td>0.389158</td>
      <td>-0.753492</td>
      <td>-0.019608</td>
      <td>-0.004902</td>
    </tr>
    <tr>
      <th>1146046</th>
      <td>2024-12-26</td>
      <td>9945</td>
      <td>潤泰新</td>
      <td>43.45</td>
      <td>43.20</td>
      <td>0.530196</td>
      <td>-0.120362</td>
      <td>0.531859</td>
      <td>0.209100</td>
      <td>-0.218872</td>
      <td>-0.003472</td>
      <td>-0.006944</td>
    </tr>
    <tr>
      <th>1146047</th>
      <td>2024-12-26</td>
      <td>9946</td>
      <td>三發地產</td>
      <td>23.70</td>
      <td>23.20</td>
      <td>0.455357</td>
      <td>-0.180938</td>
      <td>0.633320</td>
      <td>0.363988</td>
      <td>0.795507</td>
      <td>-0.030303</td>
      <td>-0.021645</td>
    </tr>
    <tr>
      <th>1146048</th>
      <td>2024-12-26</td>
      <td>9955</td>
      <td>佳龍</td>
      <td>28.70</td>
      <td>28.75</td>
      <td>0.775451</td>
      <td>-0.119026</td>
      <td>0.470710</td>
      <td>0.484027</td>
      <td>0.610103</td>
      <td>-0.017271</td>
      <td>-0.015544</td>
    </tr>
    <tr>
      <th>1146049</th>
      <td>2024-12-26</td>
      <td>9958</td>
      <td>世紀鋼</td>
      <td>174.50</td>
      <td>171.50</td>
      <td>-0.574898</td>
      <td>-1.545305</td>
      <td>0.993481</td>
      <td>0.912875</td>
      <td>0.859018</td>
      <td>-0.067055</td>
      <td>-0.043732</td>
    </tr>
  </tbody>
</table>
<p>1146050 rows × 12 columns</p>
</div>



```python
df1.to_csv('withfactor.csv',index = False)
```
