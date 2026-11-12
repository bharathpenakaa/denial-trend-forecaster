# Denial Trend Forecaster

Forecasts claim denial rates by payer and flags spikes and rising trends.

## Run
    python denial_trend_forecaster.py              # built-in sample data
    python denial_trend_forecaster.py my_claims.csv

Requires Python 3.8+. No external packages.

## CSV format
    month,payer,submitted,denied
    2026-01,Aetna,1200,130

## Output
- Ranked console report (3-month forecast, trend, spike z-score, risk)
- denial_forecasts.csv

## Note
Holt's method extrapolates the latest trend, so a one-month spike can give an
exaggerated forecast. Treat spikes as a signal to investigate.
