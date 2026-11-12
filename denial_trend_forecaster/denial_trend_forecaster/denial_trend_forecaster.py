"""
Denial Trend Forecaster
-----------------------
Forecasts future claim denial rates by payer and flags payers whose denials
are spiking or trending upward, so the revenue-cycle team can act early.

How it works
1. Input: monthly claim counts per payer (submitted, denied). Load from CSV
   or use the built-in synthetic sample data.
2. Forecast: Holt's linear exponential smoothing (level + trend). The
   smoothing parameters are tuned per payer by a small grid search that
   minimises one-step-ahead squared error.
3. Spike detection: z-score of the latest month vs. the prior 6 months.
4. Trend label: slope of a least-squares line over the last 6 months.
5. Output: ranked console report + CSV of forecasts.

Pure Python 3.8+, no external packages.

CSV format (header required):
    month,payer,submitted,denied
    2026-01,Aetna,1200,130

Run:  python denial_trend_forecaster.py
      python denial_trend_forecaster.py my_claims.csv
"""

import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

HORIZON = 3               # months to forecast
WINDOW = 6                # months used for spike / trend checks
SPIKE_Z = 2.0             # z-score above which latest month is a spike
RATE_THRESHOLD = 0.12     # forecast denial rate considered high risk
SLOPE_THRESHOLD = 0.002   # monthly change in rate counted as a real trend


# ----------------------------------------------------------------------------
# 1. Data loading / sample data
# ----------------------------------------------------------------------------
def month_add(month: str, n: int) -> str:
    y, m = int(month[:4]), int(month[5:7])
    idx = y * 12 + (m - 1) + n
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def load_csv(path: str) -> Dict[str, List[Tuple[str, int, int]]]:
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[r["payer"]].append((r["month"], int(r["submitted"]), int(r["denied"])))
    for payer in data:
        data[payer].sort()
    return data


def sample_data(months: int = 14, start: str = "2025-09", seed: int = 7):
    """Synthetic data: some payers stable, one rising, one falling, one spiking."""
    rnd = random.Random(seed)
    profiles = {
        # payer: (base rate, monthly drift, spike in last month)
        "Aetna":      (0.080, 0.000, 0.000),
        "Cigna":      (0.090, 0.006, 0.000),   # steadily rising
        "United":     (0.110, -0.003, 0.000),  # improving
        "Blue Cross": (0.070, 0.000, 0.060),   # sudden spike
        "Medicare":   (0.050, 0.001, 0.000),
    }
    data = defaultdict(list)
    for payer, (base, drift, spike) in profiles.items():
        for i in range(months):
            month = month_add(start, i)
            submitted = rnd.randint(900, 1500)
            rate = base + drift * i + rnd.uniform(-0.008, 0.008)
            if i == months - 1:
                rate += spike
            data[payer].append((month, submitted, int(submitted * max(rate, 0.0))))
    return data


# ----------------------------------------------------------------------------
# 2. Forecasting (Holt's linear exponential smoothing)
# ----------------------------------------------------------------------------
def holt_fit(series: List[float], alpha: float, beta: float):
    level, trend = series[0], series[1] - series[0]
    sse = 0.0
    residuals = []
    for y in series[1:]:
        pred = level + trend
        err = y - pred
        residuals.append(err)
        sse += err * err
        new_level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    return level, trend, sse, residuals


def holt_forecast(series: List[float], horizon: int):
    """Tune alpha/beta by grid search, return forecasts and residual std."""
    best = None
    grid = [i / 10 for i in range(1, 10)]
    for a in grid:
        for b in grid:
            level, trend, sse, res = holt_fit(series, a, b)
            if best is None or sse < best[0]:
                best = (sse, a, b, level, trend, res)
    _, a, b, level, trend, res = best
    std = math.sqrt(sum(r * r for r in res) / max(len(res), 1))
    forecasts = []
    for h in range(1, horizon + 1):
        point = min(max(level + h * trend, 0.0), 1.0)
        margin = 1.96 * std * math.sqrt(h)
        forecasts.append((point, max(point - margin, 0.0), min(point + margin, 1.0)))
    return forecasts, (a, b)


# ----------------------------------------------------------------------------
# 3. Spike and trend detection
# ----------------------------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))


def spike_zscore(rates: List[float]) -> float:
    history = rates[-(WINDOW + 1):-1]
    if len(history) < 3:
        return 0.0
    sd = stdev(history)
    if sd < 1e-9:
        return 0.0
    return (rates[-1] - mean(history)) / sd


def slope(rates: List[float]) -> float:
    ys = rates[-WINDOW:]
    n = len(ys)
    xs = list(range(n))
    xm, ym = mean(xs), mean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den else 0.0


def trend_label(s: float) -> str:
    if s > SLOPE_THRESHOLD:
        return "RISING"
    if s < -SLOPE_THRESHOLD:
        return "FALLING"
    return "STABLE"


# ----------------------------------------------------------------------------
# 4. Putting it together
# ----------------------------------------------------------------------------
@dataclass
class PayerForecast:
    payer: str
    last_month: str
    last_rate: float
    forecasts: List[Tuple[str, float, float, float]]  # (month, point, low, high)
    trend: str
    slope: float
    zscore: float
    risk: str
    note: str


def analyse_payer(payer: str, rows: List[Tuple[str, int, int]]) -> PayerForecast:
    rows = sorted(rows)
    rates = [d / s if s else 0.0 for _, s, d in rows]
    if len(rates) < 4:
        raise ValueError(f"{payer}: need at least 4 months of data")

    fc, _ = holt_forecast(rates, HORIZON)
    last_month = rows[-1][0]
    forecasts = [(month_add(last_month, i + 1), p, lo, hi) for i, (p, lo, hi) in enumerate(fc)]

    z = spike_zscore(rates)
    s = slope(rates)
    label = trend_label(s)
    peak = max(p for _, p, _, _ in forecasts)

    notes = []
    if z >= SPIKE_Z:
        notes.append("latest month is a spike")
    if label == "RISING":
        notes.append("denials trending up")
    if peak >= RATE_THRESHOLD:
        notes.append("forecast above threshold")

    if (z >= SPIKE_Z and peak >= RATE_THRESHOLD) or len(notes) >= 3:
        risk = "HIGH"
    elif notes:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return PayerForecast(payer, last_month, rates[-1], forecasts, label, s, z, risk,
                         "; ".join(notes) or "no concerns")


def run(data) -> List[PayerForecast]:
    results = [analyse_payer(p, rows) for p, rows in data.items()]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda r: (order[r.risk], -max(f[1] for f in r.forecasts)))
    return results


# ----------------------------------------------------------------------------
# 5. Output
# ----------------------------------------------------------------------------
def print_report(results: List[PayerForecast]) -> None:
    first = results[0].forecasts
    months = [f[0] for f in first]
    head = (f"{'Payer':<12}{'Last':>7}  " + "".join(f"{m:>9}" for m in months)
            + f"  {'Trend':<8}{'Z':>6}  {'Risk':<7}Notes")
    print(head)
    print("-" * (len(head) + 25))
    for r in results:
        fcs = "".join(f"{f[1] * 100:>8.1f}%" for f in r.forecasts)
        print(f"{r.payer:<12}{r.last_rate * 100:>6.1f}%  {fcs}  "
              f"{r.trend:<8}{r.zscore:>6.1f}  {r.risk:<7}{r.note}")


def export_csv(results: List[PayerForecast], path: str = "denial_forecasts.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["payer", "forecast_month", "forecast_rate", "lower_95", "upper_95",
                    "trend", "spike_z", "risk", "note"])
        for r in results:
            for month, p, lo, hi in r.forecasts:
                w.writerow([r.payer, month, round(p, 4), round(lo, 4), round(hi, 4),
                            r.trend, round(r.zscore, 2), r.risk, r.note])
    print(f"\nExported forecasts to {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = load_csv(sys.argv[1])
    else:
        print("(No CSV supplied: using synthetic sample data)\n")
        data = sample_data()
    results = run(data)
    print("Denial Rate Forecast by Payer\n")
    print_report(results)
    export_csv(results)
