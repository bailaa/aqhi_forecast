import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet

# ========== 1. Загрузка страницы и извлечение таблицы ==========
url = "https://www.aqhi.gov.hk/en/past-data/hkaq2025.html"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
resp.raise_for_status()

# pandas.read_html сам найдёт все <table> на странице
tables = pd.read_html(resp.text)

# Ищем таблицу, где есть строка PM2.5
pm_table = None
for t in tables:
    if t.astype(str).apply(lambda col: col.str.contains("PM2.5", na=False)).any().any():
        pm_table = t
        break

if pm_table is None:
    raise ValueError("Таблица с PM2.5 не найдена — проверьте структуру страницы")

# Приводим к виду year / pm25
# В таблице первая колонка — заголовки загрязнителей, дальше годы
# Найдём строку PM2.5 и годы из заголовка
pm_row = pm_table[pm_table.iloc[:, 0].astype(str).str.contains("PM2.5", na=False)].iloc[0]
years = pm_table.columns[1:]  # предполагаем, что первая колонка — названия pollutant
values = pm_row.iloc[1:].values

df = pd.DataFrame({
    "year": [int(str(y).strip()) for y in years],
    "pm25": pd.to_numeric(values, errors="coerce")
}).dropna().sort_values("year").reset_index(drop=True)

print(df)
df.to_csv("pm25_hk_2004_2025.csv", index=False)

# ========== 2. Визуализация тренда ==========
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df["pm25"], marker="o")
plt.title("Hong Kong Annual Mean PM2.5 (2004–2025)")
plt.xlabel("Year"); plt.ylabel("PM2.5 (µg/m³)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pm25_trend.png", dpi=150)

# ========== 3. Проверка гипотезы: значимый линейный тренд ==========
X = sm.add_constant(df["year"])
model = sm.OLS(df["pm25"], X).fit()
print(model.summary())  # смотрим p-value для year и R²

# ========== 4. Прогноз Prophet на 2026–2028 ==========
prophet_df = df.rename(columns={"year": "ds", "pm25": "y"})
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(str) + "-01-01")

m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=3, freq="Y")
forecast = m.predict(future)

fig = m.plot(forecast)
plt.title("PM2.5 Forecast to 2028")
plt.tight_layout()
plt.savefig("forecast.png", dpi=150)

print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5))
