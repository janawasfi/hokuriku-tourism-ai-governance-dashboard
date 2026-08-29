"""
generate_report_data.py
------------------------
Builds public/data/dashboard_data.json for the FTAS Executive Dashboard
using the project's real data pipeline (src/data_loader.py).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import load_config, resolve_repo_path
from src.report import Reporter
from src.data_loader import load_all_data

JMA_AREA_CODE = "180000"
JMA_ENDPOINT = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{JMA_AREA_CODE}.json"

RF_PARAMS = dict(
    n_estimators=500, max_depth=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)


def fetch_weather_forecast() -> list[dict]:
    """Fetch the 14-day weather forecast from the JMA Bosai API for Fukui."""
    req = urllib.request.Request(JMA_ENDPOINT, headers={"User-Agent": "FTAS-Dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"[WARN] JMA fetch failed: {e}")
        return []

    forecast_days = []
    try:
        short_term = data[0]["timeSeries"][0]
        time_defines = short_term["timeDefines"]
        weather_area = short_term["areas"][0]
        weathers = weather_area.get("weathers", [])
        winds = weather_area.get("winds", [])
        pops_series = data[0]["timeSeries"][1] if len(data[0]["timeSeries"]) > 1 else None
        pops_defines = pops_series["timeDefines"] if pops_series else []
        pops_area = pops_series["areas"][0] if pops_series else {}
        pops = pops_area.get("pops", [])

        for i, date_str in enumerate(time_defines):
            date = date_str[:10]
            pop_value = None
            for j, pdate in enumerate(pops_defines):
                if pdate[:10] == date and j < len(pops) and pops[j]:
                    pop_value = int(pops[j])
                    break
            forecast_days.append({
                "date": date,
                "weather": weathers[i] if i < len(weathers) else None,
                "wind": winds[i] if i < len(winds) else None,
                "precipitation_pct": pop_value,
                "rain_risk": bool(pop_value is not None and pop_value >= 40),
            })
    except (KeyError, IndexError, TypeError) as e:
        print(f"[WARN] Unexpected JMA response shape: {e}")

    return forecast_days


def compute_pacing_status(current_bookings: float, model_forecast: float) -> dict:
    """
    R = Current / Forecast
    Superb >= 120% | Strong 80-120% | Warning 60-80% | Critical < 60%
    """
    rate = 0.0 if model_forecast == 0 else current_bookings / model_forecast
    if rate >= 1.20:
        badge, label = "HOT", "Superb"
    elif rate >= 0.80:
        badge, label = "OK", "Strong"
    elif rate >= 0.60:
        badge, label = "WARN", "Warning"
    else:
        badge, label = "CRIT", "Critical"
    return {"rate": round(rate, 4), "badge": badge, "label": label}


def build_features(daily: pd.DataFrame, route_col: str):
    """Build calendar, weather, and RSI features on top of the merged daily table."""
    import jpholiday
    df = daily.sort_values("date").reset_index(drop=True).copy()
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["date"].apply(lambda d: int(jpholiday.is_holiday(d.date())))
    df["is_weekend_or_holiday"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    df["month"] = df["date"].dt.month
    df["weather_severity"] = (
        (df["precip"] > 0).astype(int) + (df["precip"] > 10).astype(int) + (df["wind"] > 8).astype(int)
    ).clip(upper=3)
    for lag in range(1, 4):
        df[f"{route_col}_lag{lag}"] = df[route_col].shift(lag)
    df[f"{route_col}_roll7"] = df[route_col].rolling(7, min_periods=1).mean()
    df["precip_lag1"] = df["precip"].shift(1)
    df["weekend_x_severity"] = df["is_weekend_or_holiday"] * df["weather_severity"]
    df["weekend_x_intent"] = df["is_weekend_or_holiday"] * df[route_col].fillna(0)
    feature_cols = [
        route_col, f"{route_col}_lag1", f"{route_col}_lag2", f"{route_col}_lag3",
        f"{route_col}_roll7", "precip", "temp", "sun", "wind", "precip_lag1",
        "is_weekend_or_holiday", "weather_severity", "weekend_x_severity",
        "weekend_x_intent", "month",
    ]
    return df, feature_cols


def train_and_predict(daily: pd.DataFrame, route_col: str) -> pd.DataFrame:
    """Train a Random Forest on the first 80% (chronologically) and predict on all rows."""
    df, feature_cols = build_features(daily, route_col)
    clean = df[["date", "count"] + feature_cols].dropna().reset_index(drop=True)
    split_idx = int(len(clean) * 0.80)
    train = clean.iloc[:split_idx]
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(train[feature_cols], train["count"])
    clean = clean.copy()
    clean["forecast"] = model.predict(clean[feature_cols])
    return clean[["date", "count", "forecast"]]


def build_summary(pred: pd.DataFrame) -> dict:
    """Executive summary cards: past-30-day performance vs. same period last year, plus weekly pacing."""
    last_date = pred["date"].max()

    def window_total(end_date, days):
        start = end_date - timedelta(days=days - 1)
        mask = (pred["date"] >= start) & (pred["date"] <= end_date)
        return pred.loc[mask, "count"].sum()

    current_30 = window_total(last_date, 30)
    prev_year_end = last_date - timedelta(days=365)
    previous_30 = window_total(prev_year_end, 30)
    diff = current_30 - previous_30
    yoy_pct = round((diff / previous_30 * 100), 2) if previous_30 else None
    this_week = pred[pred["date"] > last_date - timedelta(days=7)]
    this_week_pacing = compute_pacing_status(this_week["count"].sum(), this_week["forecast"].sum())

    return {
        "past_30_day": {
            "current_total": int(current_30),
            "previous_year_total": int(previous_30),
            "diff": int(diff),
            "yoy_pct": yoy_pct,
        },
        "this_week_pacing": this_week_pacing,
    }


def build_weekly_pacing(pred: pd.DataFrame) -> list[dict]:
    """Day-by-day breakdown for the most recent 14 days available (actual/forecast/achievement/badge)."""
    recent = pred.sort_values("date").tail(14)
    rows = []
    for _, r in recent.iterrows():
        status = compute_pacing_status(r["count"], r["forecast"])
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "actual": round(float(r["count"]), 1),
            "forecast": round(float(r["forecast"]), 1),
            **status,
        })
    return rows


def build_nudges(weather: list[dict], weekly_pacing: list[dict]) -> list[dict]:
    """Simple rule-based operational recommendations based on weather and the latest pacing badge."""
    nudges = []
    for day in weather:
        if day.get("rain_risk"):
            nudges.append({
                "type": "weather", "date": day["date"],
                "message": "High rain probability — activate the indoor-activity plan and notify guests in advance.",
            })
    if weekly_pacing:
        last = weekly_pacing[-1]
        if last["badge"] in ("CRIT", "WARN"):
            nudges.append({
                "type": "demand", "date": last["date"],
                "message": "Below-expected pacing — activate an urgent 5-10% OTA discount.",
            })
        elif last["badge"] == "HOT":
            nudges.append({
                "type": "demand", "date": last["date"],
                "message": "Very high demand — raise rates and enable on-site upsell offers.",
            })
    return nudges


def build_dashboard_payload(cfg: dict, reporter: Reporter) -> dict:
    """Assemble all six dashboard sections from the real data pipeline."""
    print("[1/3] Loading local data via load_all_data() ...")
    data = load_all_data(cfg, reporter)
    daily = data["daily"]
    route_col = data["route_col"]
    print(f"      -> {len(daily)} merged daily rows")

    print("[2/3] Training model and generating predictions ...")
    pred = train_and_predict(daily, route_col)

    print("[3/3] Fetching upcoming weather forecast (live JMA) ...")
    weather = fetch_weather_forecast()

    summary = build_summary(pred)
    weekly_pacing = build_weekly_pacing(pred)
    nudges = build_nudges(weather, weekly_pacing)

    demand_forecast = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "actual": round(float(r["count"]), 1),
            "forecast": round(float(r["forecast"]), 1),
        }
        for _, r in pred.sort_values("date").tail(60).iterrows()
    ]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "weather_strip": weather,
        "demand_forecast": demand_forecast,
        "weekly_pacing": weekly_pacing,
        "nudges": nudges,
    }


def export_json(payload: dict, cfg: dict) -> None:
    output_path = resolve_repo_path(cfg, "public", "data", "dashboard_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] dashboard_data.json written to {output_path}")


def main():
    cfg = load_config()
    reporter = Reporter(cfg)
    payload = build_dashboard_payload(cfg, reporter)
    export_json(payload, cfg)


if __name__ == "__main__":
    main()
