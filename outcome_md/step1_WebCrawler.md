```python
import pandas as pd
import numpy as np
import json
import requests
import time
```

```python
def daterange(start , end):
    d0 = pd.to_datetime(start)
    d1 = pd.to_datetime(end)
    return [d.strftime("%Y%m%d") for d in pd.date_range(d0, d1, freq = "D")]
```

```python
COLS = ["日期","代號","名稱","成交股數","成交筆數","成交金額","開盤價","最高價","最低價","收盤價"]
NUM_COLS = ["成交股數","成交筆數","成交金額","開盤價","最高價","最低價","收盤價"]

def data_clean(data_list):
    out = []
    for date_raw, rows in data_list:
        date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        rows9 = [r[:9] for r in rows if isinstance(r, list) and len(r) >= 9 and isinstance(r[0], str)]
        start = next((i for i, r in enumerate(rows9) if r[0].strip() == "1101"), None)
        out.extend([[date] + r for r in rows9[start:]])

    df = pd.DataFrame(out, columns=COLS)
    
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
        
    return df.sort_values(["日期","代號"], ignore_index=True)
```

```python
start = ['20200101','20230101']
end = ['20221231','20241231']

df = pd.DataFrame()
for i in range(len(start)):
    data_list = []
    for date in daterange(start[i] , end[i]):
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?response=json&type=ALLBUT0999&date={date}"
        data = requests.get(url).text
        data = json.loads(data)
        if data == {'stat': '很抱歉，沒有符合條件的資料!'}:
            time.sleep(1 + np.random.uniform(0, 1))
            continue
        else:
            need = [row[:9] for row in (data["tables"][8]['data'])]
            data_list.append([date,need])
            time.sleep(2 + np.random.uniform(0, 1))
    df = pd.concat([df,data_clean(data_list)])
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
      <td>1101B</td>
      <td>台泥乙特</td>
      <td>12000</td>
      <td>10</td>
      <td>643200</td>
      <td>53.50</td>
      <td>53.80</td>
      <td>53.50</td>
      <td>53.80</td>
    </tr>
    <tr>
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>496953</th>
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
      <th>496954</th>
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
      <th>496955</th>
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
      <th>496956</th>
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
      <th>496957</th>
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
<p>1220672 rows × 10 columns</p>
</div>



```python
mask_stock = df["代號"].astype(str).str.strip().str.fullmatch(r"\d{4}")
df = df[mask_stock]
```

```python
df.to_csv("WebData.csv", index = False)
```
