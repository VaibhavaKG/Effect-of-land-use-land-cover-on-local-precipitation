import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, gamma, kstest, t, spearmanr
from numpy.linalg import lstsq

#  Consistent style 
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "text.color":       "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
})

ACCENT   = "#58a6ff"   # blue
ACCENT2  = "#f78166"   # orange-red
ACCENT3  = "#3fb950"   # green
ACCENT4  = "#d2a8ff"   # purple
CITY_PALETTE = [
    "#58a6ff", "#f78166", "#3fb950", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff7b72",
]

# Load data 
ds   = xr.open_dataset("yearly_sum.nc")
rain = ds['RAINFALL']

# City coordinates 
cities = {
    "Delhi":     (28.6139, 77.2090),
    "Jaipur":    (26.9124, 75.7873),
    "Kolkata":   (22.5726, 88.3639),
    "Indore":    (22.7196, 75.8577),
    "Mumbai":    (19.0760, 72.8777),
    "Pune":      (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
}

# Helper: extract rainfall + year array for a city 
def get_city_series(lat, lon):
    """Returns (years_array, rainfall_array) with NaNs removed."""
    point = rain.sel(LATITUDE=lat, LONGITUDE=lon, method='nearest')
    years = point['YEAR'].values if 'YEAR' in point.coords else np.arange(len(point))
    vals  = point.values
    mask  = ~np.isnan(vals)
    return years[mask].astype(float), vals[mask]

def get_city_data(lat, lon):
    """Returns only the rainfall values (backward-compat)."""
    _, data = get_city_series(lat, lon)
    return data


# 1.  DISTRIBUTION PLOT — all cities
fig, axes = plt.subplots(3, 3, figsize=(17, 14))
fig.patch.set_facecolor("#0f1117")
axes = axes.flatten()

all_data  = []
valid_idx = 0

for (city, (lat, lon)), color in zip(cities.items(), CITY_PALETTE):
    data = get_city_data(lat, lon)
    if len(data) < 5:
        continue
    all_data.extend(data)

    mu, sigma          = norm.fit(data)
    shape, loc, scale  = gamma.fit(data, floc=0)
    x = np.linspace(min(data), max(data), 200)

    ax = axes[valid_idx]
    ax.hist(data, bins="auto", density=True, alpha=0.45, color=color, edgecolor="none")
    ax.plot(x, norm.pdf(x, mu, sigma),        color=ACCENT,  linewidth=2,   label="Normal")
    ax.plot(x, gamma.pdf(x, shape, loc, scale), color=ACCENT2, linewidth=2, label="Gamma")
    ax.set_title(city, fontsize=11, fontweight="bold", color="#e6edf3")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    valid_idx += 1

for j in range(valid_idx, len(axes)):
    fig.delaxes(axes[j])

handles = [
    plt.Line2D([0], [0], color=ACCENT,  linewidth=2, label="Normal fit"),
    plt.Line2D([0], [0], color=ACCENT2, linewidth=2, label="Gamma fit"),
]
fig.legend(handles, ["Normal fit", "Gamma fit"],
           loc="lower right", fontsize=10, framealpha=0.6)
fig.suptitle("Rainfall Distribution Across Cities", fontsize=20, fontweight="bold",
             color="#e6edf3", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("01_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# 2.  COMBINED STATISTICS + KS TEST

all_data = np.array(all_data)
mean_all = np.mean(all_data)
var_all  = np.var(all_data)

print("\n" + "═"*52)
print("  COMBINED STATISTICS")
print("═"*52)
print(f"  Mean     : {mean_all:.2f} mm")
print(f"  Variance : {var_all:.2f} mm²")

mu, sigma         = norm.fit(all_data)
shape, loc, scale = gamma.fit(all_data, floc=0)
ks_norm  = kstest(all_data, 'norm',  args=(mu, sigma))
ks_gamma = kstest(all_data, 'gamma', args=(shape, loc, scale))

print(f"\n  KS Statistic — Normal : {ks_norm.statistic:.4f}  (p={ks_norm.pvalue:.4f})")
print(f"  KS Statistic — Gamma  : {ks_gamma.statistic:.4f}  (p={ks_gamma.pvalue:.4f})")
print(f"  Better fit            : {'Gamma' if ks_gamma.statistic < ks_norm.statistic else 'Normal'}")


# 3.  DATA SMOOTHING
smoothed = np.convolve(all_data, np.ones(3)/3, mode='valid')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(all_data,  color=ACCENT,  linewidth=1.4, alpha=0.6, label="Original")
ax.plot(smoothed,  color=ACCENT2, linewidth=2.0, label="Smoothed (3-pt MA)")
ax.set_title("Data Smoothing — Moving Average", fontsize=14, fontweight="bold")
ax.set_xlabel("Index"); ax.set_ylabel("Rainfall (mm)")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("02_smoothing.png", dpi=150, bbox_inches="tight")
plt.show()


# 4.  TIME SERIES — all combined

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(all_data, color=ACCENT, linewidth=1.5)
ax.set_title("Rainfall Time Series (All Cities Combined)", fontsize=14, fontweight="bold")
ax.set_xlabel("Index"); ax.set_ylabel("Rainfall (mm)")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("03_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()


# 5.  COMBINED DISTRIBUTION — Normal fit per city

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(all_data, bins="auto", density=True, alpha=0.3,
        color="#8b949e", edgecolor="none", label="All Cities Combined")

for (city, (lat, lon)), color in zip(cities.items(), CITY_PALETTE):
    data = get_city_data(lat, lon)
    if len(data) < 5:
        continue
    mu_c, sigma_c = norm.fit(data)
    x_c = np.linspace(min(data), max(data), 200)
    ax.plot(x_c, norm.pdf(x_c, mu_c, sigma_c), linewidth=1.8, color=color, label=city)

ax.set_title("Combined Rainfall Distribution — Normal Fit per City", fontsize=14, fontweight="bold")
ax.set_xlabel("Rainfall (mm)"); ax.set_ylabel("Density")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("04_combined_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# 6.  HYPOTHESIS TESTING — Coastal vs Inland

from scipy.stats import ttest_ind

coastal = ["Mumbai", "Kolkata"]
inland  = ["Delhi", "Jaipur", "Indore", "Pune", "Hyderabad", "Bengaluru"]

high_data, low_data = [], []
for city, (lat, lon) in cities.items():
    data = get_city_data(lat, lon)
    if len(data) < 5:
        continue
    if city in coastal:
        high_data.extend(data)
    elif city in inland:
        low_data.extend(data)

high_data = np.array(high_data)
low_data  = np.array(low_data)

t_stat, p_value = ttest_ind(high_data, low_data, equal_var=False)
alpha = 0.05

print("\n" + "═"*52)
print("  HYPOTHESIS TESTING (Coastal vs Inland)")
print("═"*52)
print("  H0: Mean rainfall (Coastal) = Mean rainfall (Inland)")
print("  H1: Mean rainfall (Coastal) ≠ Mean rainfall (Inland)")
print(f"\n  n Coastal  : {len(high_data)}")
print(f"  n Inland   : {len(low_data)}")
print(f"  Mean Coastal : {np.mean(high_data):.2f} mm")
print(f"  Mean Inland  : {np.mean(low_data):.2f} mm")
print(f"  t-statistic  : {t_stat:.4f}")
print(f"  p-value      : {p_value:.6f}")
print(f"\n  Decision (α=0.05): {'Reject H0 → Significant difference' if p_value < alpha else 'Fail to reject H0'}")
print("  Conclusion: Coastal cities show significantly higher rainfall than inland cities")


# 7.  REGRESSION — Year as predictor  (TREND ANALYSIS)
print("\n" + "═"*52)
print("  REGRESSION — TEMPORAL TREND ANALYSIS")
print("  Predictor: Year  |  Response: Annual Rainfall")
print("═"*52)

# 7A. Linear Trend — all cities on one plot 
fig, ax = plt.subplots(figsize=(12, 6))

for (city, (lat, lon)), color in zip(cities.items(), CITY_PALETTE):
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue
    coeffs  = np.polyfit(years, data, 1)
    y_fit   = np.polyval(coeffs, years)
    ax.scatter(years, data, color=color, s=18, alpha=0.5, zorder=3)
    ax.plot(years, y_fit, color=color, linewidth=2,
            label=f"{city}  ({coeffs[0]:+.1f} mm/yr)")

ax.set_title("Linear Trend: Annual Rainfall vs Year (per city)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Rainfall (mm)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("05A_linear_trend.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n  [7A] Linear Trend Slopes (mm / year)")
print(f"  {'City':<12}  {'Slope':>10}  {'Intercept':>12}  {'R²':>7}")
print("  " + "-"*46)
for city, (lat, lon) in cities.items():
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue
    coeffs  = np.polyfit(years, data, 1)
    y_pred  = np.polyval(coeffs, years)
    ss_res  = np.sum((data - y_pred)**2)
    ss_tot  = np.sum((data - np.mean(data))**2)
    r2      = 1 - ss_res / ss_tot
    direction = "↑ Increasing" if coeffs[0] > 0 else "↓ Decreasing"
    print(f"  {city:<12}  {coeffs[0]:>+10.2f}  {coeffs[1]:>12.2f}  {r2:>7.3f}  {direction}")


# 7B. Polynomial (deg-2) Trend per city — subplot grid 
fig, axes = plt.subplots(3, 3, figsize=(17, 14))
fig.patch.set_facecolor("#0f1117")
axes = axes.flatten()
valid_idx = 0

for (city, (lat, lon)), color in zip(cities.items(), CITY_PALETTE):
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue

    coeffs   = np.polyfit(years, data, 2)
    x_fit    = np.linspace(years.min(), years.max(), 200)
    y_fit    = np.polyval(coeffs, x_fit)
    y_pred   = np.polyval(coeffs, years)
    r2       = 1 - np.sum((data - y_pred)**2) / np.sum((data - np.mean(data))**2)

    ax = axes[valid_idx]
    ax.scatter(years, data, color=color, s=20, alpha=0.55, zorder=3)
    ax.plot(x_fit, y_fit, color=ACCENT2, linewidth=2,
            label=f"Poly deg-2 | R²={r2:.3f}")
    ax.set_title(city, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Rainfall (mm)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    valid_idx += 1

for j in range(valid_idx, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Polynomial (deg-2) Rainfall Trend — per City", fontsize=18,
             fontweight="bold", color="#e6edf3", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("05B_poly_trend.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n  [7B] Polynomial (deg-2) Trend R² values")
print(f"  {'City':<12}  {'R²':>7}")
print("  " + "-"*22)
for city, (lat, lon) in cities.items():
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue
    coeffs = np.polyfit(years, data, 2)
    y_pred = np.polyval(coeffs, years)
    r2     = 1 - np.sum((data - y_pred)**2) / np.sum((data - np.mean(data))**2)
    print(f"  {city:<12}  {r2:>7.3f}")


# 7C. Power-Law Trend using (Year − base_year) as the x-axis 

fig, axes = plt.subplots(3, 3, figsize=(17, 14))
fig.patch.set_facecolor("#0f1117")
axes = axes.flatten()
valid_idx = 0

print("\n  [7C] Power-Law Trend  y = a · t^b  (t = elapsed years)")
print(f"  {'City':<12}  {'a':>10}  {'b':>8}  {'R²':>7}")
print("  " + "-"*42)

for (city, (lat, lon)), color in zip(cities.items(), CITY_PALETTE):
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue

    t_elapsed = years - years.min() + 1        # shift so t starts at 1
    log_t     = np.log(t_elapsed)
    log_y     = np.log(np.maximum(data, 1e-6)) # guard against zeros

    coeffs    = np.polyfit(log_t, log_y, 1)
    b         = coeffs[0]
    a         = np.exp(coeffs[1])

    t_fit     = np.linspace(t_elapsed.min(), t_elapsed.max(), 200)
    y_fit     = a * t_fit ** b

    log_y_pred = np.polyval(coeffs, log_t)
    r2         = 1 - np.sum((log_y - log_y_pred)**2) / np.sum((log_y - np.mean(log_y))**2)

    print(f"  {city:<12}  {a:>10.2f}  {b:>8.4f}  {r2:>7.3f}")

    ax = axes[valid_idx]
    ax.scatter(years, data, color=color, s=20, alpha=0.55, zorder=3)
    ax.plot(years.min() + t_fit - 1, y_fit, color=ACCENT3, linewidth=2,
            label=f"y={a:.0f}·t^{b:.2f} | R²={r2:.3f}")
    ax.set_title(city, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Rainfall (mm)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    valid_idx += 1

for j in range(valid_idx, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Power-Law Rainfall Trend — per City", fontsize=18,
             fontweight="bold", color="#e6edf3", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("05C_powerlaw_trend.png", dpi=150, bbox_inches="tight")
plt.show()


# 7D. Multiple Linear Regression — Year + Latitude 
#   For each city we have one data point per year.
#   We build a pooled dataset: response = rainfall, predictors = [year, latitude].

print("\n  [7D] Multiple Linear Regression — Predictors: Year + Latitude")

X1_all, X2_all, Y_all, city_labels = [], [], [], []
for city, (lat, lon) in cities.items():
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue
    X1_all.extend(years)
    X2_all.extend([lat] * len(years))
    Y_all.extend(data)
    city_labels.extend([city] * len(years))

X1_all = np.array(X1_all)
X2_all = np.array(X2_all)
Y_all  = np.array(Y_all)

A      = np.column_stack([np.ones(len(Y_all)), X1_all, X2_all])
coeffs_mlr, _, _, _ = lstsq(A, Y_all, rcond=None)
b0, b1, b2 = coeffs_mlr
Y_pred_mlr = A @ coeffs_mlr

ss_res = np.sum((Y_all - Y_pred_mlr)**2)
ss_tot = np.sum((Y_all - np.mean(Y_all))**2)
r2_mlr = 1 - ss_res / ss_tot

print(f"\n  Intercept  (β₀) : {b0:.2f}")
print(f"  Year       (β₁) : {b1:.4f}  mm per year")
print(f"  Latitude   (β₂) : {b2:.4f}  mm per degree")
print(f"  R²              : {r2_mlr:.4f}")

fig, ax = plt.subplots(figsize=(9, 6))
scatter_colors = [CITY_PALETTE[list(cities.keys()).index(c)] for c in city_labels]
ax.scatter(Y_all, Y_pred_mlr, c=scatter_colors, s=15, alpha=0.4, zorder=3)
ax.plot([Y_all.min(), Y_all.max()], [Y_all.min(), Y_all.max()],
        color=ACCENT2, linewidth=2, linestyle="--", label="Perfect fit")

# City legend patches
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=CITY_PALETTE[i], label=city)
           for i, city in enumerate(cities.keys())]
ax.legend(handles=patches, fontsize=8, loc="upper left", ncol=2)
ax.set_title(f"Multiple Linear Regression  (Year + Latitude → Rainfall)\nR² = {r2_mlr:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Rainfall (mm)"); ax.set_ylabel("Predicted Rainfall (mm)")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("05D_multiple_regression.png", dpi=150, bbox_inches="tight")
plt.show()


# 7E. Spearman Rank Correlation — Year vs Rainfall per city 
print("\n  [7E] Spearman Rank Correlation — Year vs Rainfall")
print(f"  {'City':<12}  {'ρ':>8}  {'p-value':>10}  Interpretation")
print("  " + "-"*58)

fig, axes = plt.subplots(3, 3, figsize=(17, 14))
fig.patch.set_facecolor("#0f1117")
axes = axes.flatten()
valid_idx = 0

for (city, (lat, lon)), color in zip(cities.items(), CITY_PALETTE):
    years, data = get_city_series(lat, lon)
    if len(data) < 5:
        continue

    rho, p_val = spearmanr(years, data)
    sig = "Significant" if p_val < 0.05 else "Not significant"
    direction = ("↑ Increasing" if rho > 0 else "↓ Decreasing") if p_val < 0.05 else "→ No trend"
    print(f"  {city:<12}  {rho:>8.4f}  {p_val:>10.4f}  {sig} {direction}")

    ax = axes[valid_idx]
    ax.scatter(years, data, color=color, s=20, alpha=0.55, zorder=3)

    # Overlay the linear trend line for visual reference
    coeffs = np.polyfit(years, data, 1)
    ax.plot(years, np.polyval(coeffs, years), color=ACCENT4, linewidth=2,
            label=f"ρ={rho:.3f}  p={p_val:.4f}")
    ax.set_title(city, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Rainfall (mm)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    valid_idx += 1

for j in range(valid_idx, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Spearman Rank Correlation: Year vs Rainfall — per City",
             fontsize=18, fontweight="bold", color="#e6edf3", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("05E_spearman_trend.png", dpi=150, bbox_inches="tight")
plt.show()


# 8.  CONFIDENCE INTERVALS
# 8.1 Analytical CI
n      = len(all_data)
mean   = np.mean(all_data)
std    = np.std(all_data, ddof=1)
t_crit = t.ppf(0.975, df=n-1)
margin = t_crit * std / np.sqrt(n)

print("\n" + "═"*52)
print("  CONFIDENCE INTERVALS")
print("═"*52)
print(f"\n  [8.1] Analytical 95% CI")
print(f"  Mean   : {mean:.2f} mm")
print(f"  95% CI : [{mean - margin:.2f},  {mean + margin:.2f}]")

# 8.2 Bootstrap CI
n_sim   = 1000
means_b = [np.mean(np.random.choice(all_data, size=n, replace=True))
           for _ in range(n_sim)]
means_b = np.array(means_b)
lower_b = np.percentile(means_b, 2.5)
upper_b = np.percentile(means_b, 97.5)

print(f"\n  [8.2] Bootstrap Monte-Carlo 95% CI  (n_sim={n_sim})")
print(f"  Mean estimate : {np.mean(means_b):.2f} mm")
print(f"  95% CI        : [{lower_b:.2f},  {upper_b:.2f}]")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(means_b, bins=35, density=True, alpha=0.55, color=ACCENT, edgecolor="none")
ax.axvline(lower_b, color=ACCENT2, linewidth=2, linestyle="--", label=f"Lower CI  {lower_b:.2f}")
ax.axvline(upper_b, color=ACCENT3, linewidth=2, linestyle="--", label=f"Upper CI  {upper_b:.2f}")
ax.axvline(np.mean(means_b), color="white", linewidth=1.5, linestyle=":", label=f"Mean  {np.mean(means_b):.2f}")
ax.set_title("Bootstrap Monte-Carlo — 95% CI of Mean Rainfall", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Rainfall (mm)"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("06_bootstrap_ci.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "═"*52)
print("  Analysis complete.  All figures saved.")
print("═"*52)
