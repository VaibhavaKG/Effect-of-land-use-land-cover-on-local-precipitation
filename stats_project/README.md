# PHY 5132 — Statistics Course Project
## Rainfall Analysis over Built-up Areas | 2005–2017

---

## About This Module

This module covers the **statistical analysis of mean rainfall** across 8 cities and 9 states in India over the period 2005–2017, using both annual and JJAS (June–July–August–September) monsoon season data.

> Note: This is one part of a larger group project. Other modules are handled separately by other contributors.

---

## Data Used

| File | Description |
|------|-------------|
| `yearly_mean_rainfall_builtup_cities.csv` | Annual mean rainfall for 8 cities |
| `yearly_mean_rainfall_builtup_states.csv` | Annual mean rainfall for 9 states |
| `jjas_mean_rain_cities.csv` | JJAS seasonal mean rainfall for 8 cities |
| `jjas_mean_rain_states.csv` | JJAS seasonal mean rainfall for 9 states |

**Cities:** Ahmadabad, Bengaluru, Hyderabad, Indore, Jaipur, Nagpur, Pimpri Chinchwad, Pune

**States:** Andhra Pradesh, Bihar, Chhattisgarh, Himachal Pradesh, Jammu & Kashmir, Madhya Pradesh, Meghalaya, Tamil Nadu, Telangana, Uttar Pradesh, Uttarakhand

---

## Files in This Module

```
stats_project/
├── rainfall_analysis_updated.py                          ← main analysis script
├── rainfall_analysis_updated.ipynb                       ← Jupyter notebook version
├── yearly_mean_rainfall_builtup_cities.csv
├── yearly_mean_rainfall_builtup_states.csv
├── jjas_mean_rain_cities.csv
├── jjas_mean_rain_states.csv
└── README.md
```

---

## What the Code Does

### Section 3 — Normal Distribution Fitting (Annual)
Fits a Normal distribution N(μ, σ) to the annual rainfall of each city and state. A KS (Kolmogorov–Smirnov) test is run to evaluate goodness of fit. Results are printed as a summary table with a ✓ Good / ✗ Poor label at α = 0.05.

**Output:** `fig_3A_normal_cities.png`, `fig_3B_normal_states.png`

---

### Section 4 — Gamma Distribution Fitting (Annual)
Fits a Gamma distribution Gamma(k, θ) with loc fixed at 0. KS test is again used for evaluation. Gamma is expected to outperform Normal for high-rainfall orographic locations due to its positive skew.

**Output:** `fig_4A_gamma_cities.png`, `fig_4B_gamma_states.png`

---

### Section 5 — Data Smoothing (Annual)
Applies a 3-year and 5-year centred moving average to the annual rainfall series of each region to isolate short-term and long-term trends from noise.

**Output:** `fig_5A_smoothing_cities.png`, `fig_5B_smoothing_states.png`

---

### Section 6 — JJAS Time Series
Plots the monsoon season (JJAS) rainfall time series for all cities and states together on a single panel each, covering 2005–2017.

**Output:** `fig_6A_timeseries_jjas_cities.png`, `fig_6B_timeseries_jjas_states.png`

---

### Section 7 — Hypothesis Testing (JJAS)
Tests whether mean JJAS rainfall differs significantly between two climatologically distinct groups using:
- **Welch's t-test** (parametric, unequal variance)
- **Mann-Whitney U test** (non-parametric, distribution-free)

Significance level α = 0.05, two-tailed. Results are printed to console — no plots.

**Cities:** High-rainfall group (Nagpur, Pune, Pimpri Chinchwad, Indore) vs Low-rainfall group (Ahmadabad, Bengaluru, Hyderabad, Jaipur)

**States:** High-rainfall group (Meghalaya, Chhattisgarh, Bihar) vs Low-rainfall group (Tamil Nadu, Jammu & Kashmir, Himachal Pradesh)

---

## Libraries In-Use

```
numpy
pandas
matplotlib
scipy
```
