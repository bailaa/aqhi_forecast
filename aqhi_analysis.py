import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet

URL = "https://www.aqhi.gov.hk/en/past-data/hkaq2025.html"

# Данные со страницы EPD (2004–2025) — запасной вариант, если парсинг не сработает
FALLBACK_PM25 = {
    "year": list(range(2004, 2026)),
    "pm25": [44.4, 40.7, 39.6, 40.2, 37.6, 30.9, 29.3, 33.0, 27.5, 31.0,
             28.7, 25.3, 22.0, 21.8, 19.9, 19.2, 15.0, 15.2, 14.0, 14.6, 15.1, 14.6],
}


def load_data() -> pd.DataFrame:
    """Пытается спарсить таблицу со страницы EPD, иначе берёт запасные данные."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)

        for t in tables:
            # Ищем строку PM2.5
            mask = t.iloc[:, 0].astype(str).str.contains("PM2.5", na=False)
            if mask.any():
                row = t[mask].iloc[0]
                years = [int(str(y).strip()) for y in t.columns[1:]]
                values = pd.to_numeric(row.iloc[1:].values, errors="coerce")
                df = pd.DataFrame({"year": years, "pm25": values}).dropna()
                if len(df) >= 10:
                    print(f"[OK] Спарсено {len(df)} строк со страницы EPD")
                    return df.sort_values("year").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] Парсинг не удался: {e}")

    print("[INFO] Использую запасные данные из отчёта EPD")
    return pd.DataFrame(FALLBACK_PM25)


def main():
    df = load_data()
    print(df.tail())
    df.to_csv("pm25_hk_2004_2025.csv", index=False)

    # График тренда
    plt.figure(figsize=(10, 5))
    plt.plot(df["year"], df["pm25"], marker="o")
    plt.title("Hong Kong Annual Mean PM2.5 (2004–2025)")
    plt.xlabel("Year"); plt.ylabel("PM2.5 (µg/m³)")
    plt.grid(True); plt.tight_layout()
    plt.savefig("pm25_trend.png", dpi=150)

    # Гипотеза: значимый линейный тренд
    X = sm.add_constant(df["year"])
    model = sm.OLS(df["pm25"], X).fit()
    print(model.summary())

    # Прогноз Prophet
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


if __name__ == "__main__":
    main()
