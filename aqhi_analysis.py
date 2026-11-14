import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet
from io import StringIO
from scipy.stats import pearsonr, spearmanr

# ========== КОНСТАНТЫ ==========
URL = "https://www.aqhi.gov.hk/en/past-data/hkaq2025.html"

FALLBACK_PM25 = {
    "year": list(range(2004, 2026)),
    "pm25": [44.4, 40.7, 39.6, 40.2, 37.6, 30.9, 29.3, 33.0, 27.5, 31.0,
             28.7, 25.3, 22.0, 21.8, 19.9, 19.2, 15.0, 15.2, 14.0, 14.6, 15.1, 14.6],
}

FALLBACK_LOW_VIS = {
    "year": list(range(2004, 2026)),
    "low_vis": [1570, 1502, 1257, 1298, 1100, 1139, 1225, 1399, 887, 978,
                622, 667, 509, 645, 518, 475, 346, 258, 401, 296, 286, 266],
}

# Имена файлов для сохранения
FILE_DATA_CSV = "pm25_hk_2004_2025.csv"
FILE_TREND_PNG = "pm25_trend.png"
FILE_SCATTER_PNG = "pm25_vs_visibility.png"
FILE_FORECAST_TREND_PNG = "forecast_2029_trend.png"
FILE_FORECAST_POINTS_PNG = "forecast_2029_points.png"
FILE_BAR_PNG = "pm25_2020_2029_bar.png"


# ========== ЗАГРУЗКА ДАННЫХ ==========
def load_data() -> pd.DataFrame:
    """Пытается спарсить PM2.5 со страницы EPD, иначе fallback."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        for t in tables:
            mask = t.iloc[:, 0].astype(str).str.contains("PM2.5", na=False)
            if mask.any():
                row = t[mask].iloc[0]
                years = [int(str(y).strip()) for y in t.columns[1:]]
                values = pd.to_numeric(row.iloc[1:].values, errors="coerce")
                df = pd.DataFrame({"year": years, "pm25": values}).dropna()
                if len(df) >= 10:
                    print(f"[OK] Спарсено {len(df)} строк PM2.5")
                    return df.sort_values("year").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] Парсинг PM2.5 не удался: {e}")

    print("[INFO] Использую запасные данные PM2.5")
    return pd.DataFrame(FALLBACK_PM25)


def load_low_vis() -> pd.DataFrame:
    """Пытается спарсить low visibility, иначе fallback."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        for t in tables:
            mask = t.iloc[:, 0].astype(str).str.contains("Low Visibility", na=False)
            if mask.any():
                row = t[mask].iloc[0]
                years = [int(str(y).strip()) for y in t.columns[1:]]
                values = pd.to_numeric(row.iloc[1:].values, errors="coerce")
                df = pd.DataFrame({"year": years, "low_vis": values}).dropna()
                if len(df) >= 10:
                    print(f"[OK] Спарсено {len(df)} строк low visibility")
                    return df.sort_values("year").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] Парсинг low visibility не удался: {e}")

    print("[INFO] Использую запасные данные low visibility")
    return pd.DataFrame(FALLBACK_LOW_VIS)


def prepare_data() -> pd.DataFrame:
    """Загружает и объединяет оба ряда по году."""
    df = load_data().merge(load_low_vis(), on="year", how="inner")
    print(df.tail())
    df.to_csv(FILE_DATA_CSV, index=False)
    print(f"[OK] Данные сохранены в {FILE_DATA_CSV}")
    return df


# ========== АНАЛИЗ ==========
def run_ols_trend(df: pd.DataFrame):
    """OLS-регрессия: зависимость PM2.5 от года. Возвращает результат модели."""
    X = sm.add_constant(df["year"])
    model = sm.OLS(df["pm25"], X).fit()
    print("\n===== OLS: тренд PM2.5 =====")
    print(model.summary())
    return model


def run_correlation(df: pd.DataFrame):
    """Корреляция PM2.5 и low visibility. Возвращает (r_pearson, p_pearson)."""
    r_p, p_p = pearsonr(df["pm25"], df["low_vis"])
    r_s, p_s = spearmanr(df["pm25"], df["low_vis"])
    print("\n===== Корреляция PM2.5 и Low Visibility =====")
    print(f"Pearson  r={r_p:.3f}, p={p_p:.4g}")
    print(f"Spearman r={r_s:.3f}, p={p_s:.4g}")
    return r_p, p_p


def fit_prophet(df: pd.DataFrame, periods: int = 5) -> tuple[Prophet, pd.DataFrame]:
    """Обучает Prophet на PM2.5 и возвращает модель + прогноз на `periods` лет вперёд."""
    prophet_df = df[["year", "pm25"]].rename(columns={"year": "ds", "pm25": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(str) + "-01-01")

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods, freq="YE")
    forecast = m.predict(future)

    print("\n===== Прогноз Prophet =====")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods + 1))
    return m, forecast


# ========== ГРАФИКИ ==========
def plot_trend(df: pd.DataFrame):
    """График исторического тренда PM2.5."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["year"], df["pm25"], marker="o")
    plt.title("Hong Kong Annual Mean PM2.5 (2004–2025)")
    plt.xlabel("Year")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FILE_TREND_PNG, dpi=150)
    plt.close()
    print(f"[OK] Сохранён {FILE_TREND_PNG}")


def plot_scatter(df: pd.DataFrame, r_pearson: float):
    """Scatter plot PM2.5 vs Low Visibility с линией тренда."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df["pm25"], df["low_vis"], color="steelblue", s=60, alpha=0.8)
    z = np.polyfit(df["pm25"], df["low_vis"], 1)
    p = np.poly1d(z)
    plt.plot(df["pm25"], p(df["pm25"]), "r--", alpha=0.8, label="Trend line")
    plt.xlabel("PM2.5 (µg/m³)", fontsize=12)
    plt.ylabel("Low visibility hours", fontsize=12)
    plt.title(f"PM2.5 vs Low Visibility (Pearson r={r_pearson:.2f})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FILE_SCATTER_PNG, dpi=150)
    plt.close()
    print(f"[OK] Сохранён {FILE_SCATTER_PNG}")


def plot_forecast_trend(m: Prophet, forecast: pd.DataFrame):
    """Стандартный график Prophet с прогнозом."""
    fig = m.plot(forecast, figsize=(10, 6))
    plt.title("PM2.5 Forecast up to 2029", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(FILE_FORECAST_TREND_PNG, dpi=150)
    plt.close()
    print(f"[OK] Сохранён {FILE_FORECAST_TREND_PNG}")


def plot_forecast_points(df: pd.DataFrame, forecast: pd.DataFrame):
    """Точечный график: исторические точки + линия прогноза + 95% CI."""
    forecast_years = forecast["ds"].dt.year
    plt.figure(figsize=(10, 6))
    plt.scatter(df["year"], df["pm25"], color="black", label="Historical data", zorder=3)
    plt.plot(forecast_years, forecast["yhat"], color="tab:blue", label="Forecast", zorder=2)
    plt.fill_between(forecast_years, forecast["yhat_lower"], forecast["yhat_upper"],
                     color="tab:blue", alpha=0.2, label="95% CI", zorder=1)
    plt.title("PM2.5: Historical Data & Forecast to 2029", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("PM2.5 (µg/m³)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FILE_FORECAST_POINTS_PNG, dpi=150)
    plt.close()
    print(f"[OK] Сохранён {FILE_FORECAST_POINTS_PNG}")


def plot_bar_2020_2029(df: pd.DataFrame, forecast: pd.DataFrame):
    """Bar chart: исторические 2020–2025 + прогноз 2026–2029."""
    hist = df[df["year"] >= 2020][["year", "pm25"]].rename(columns={"pm25": "value"})
    hist["type"] = "Historical"

    fc = forecast[forecast["ds"].dt.year >= 2026][["ds", "yhat"]].copy()
    fc["year"] = fc["ds"].dt.year
    fc = fc[fc["year"] <= 2029].rename(columns={"yhat": "value"})[["year", "value"]]
    fc["type"] = "Forecast"

    bar_data = pd.concat([hist, fc]).sort_values("year")

    plt.figure(figsize=(10, 6))
    colors = ["steelblue" if t == "Historical" else "orange" for t in bar_data["type"]]
    bars = plt.bar(bar_data["year"].astype(str), bar_data["value"],
                   color=colors, edgecolor="black", alpha=0.85)

    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.3,
                 round(y, 1), ha="center", va="bottom", fontsize=10)

    plt.title("PM2.5 by Year: 2020–2029 (Historical vs Forecast)", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("PM2.5 (µg/m³)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Historical (2020–2025)"),
        Patch(facecolor="orange", edgecolor="black", label="Forecast (2026–2029)"),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(FILE_BAR_PNG, dpi=150)
    plt.close()
    print(f"[OK] Сохранён {FILE_BAR_PNG}")


# ========== ГЛАВНАЯ ФУНКЦИЯ ==========
def main():
    df = prepare_data()

    run_ols_trend(df)
    r_p, _ = run_correlation(df)

    m, forecast = fit_prophet(df, periods=5)  # прогноз до 2029

    plot_trend(df)
    plot_scatter(df, r_p)
    plot_forecast_trend(m, forecast)
    plot_forecast_points(df, forecast)
    plot_bar_2020_2029(df, forecast)

    print("\nВсе готово. Файлы в папке проекта.")


if __name__ == "__main__":
    main()


    
