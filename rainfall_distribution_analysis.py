import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, kstest


ds = xr.open_dataset("yearly_sum.nc")
rain = ds['RAINFALL']


# 1. City coordinates
cities = {
    "Ahmedabad": (23.0225, 72.5714),
    "Bengaluru": (12.9716, 77.5946),
    "Meghalaya": (25.4670, 91.3662),
    "Bihar": (25.0961, 85.3131),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Indore": (22.7196, 75.8577),
    "Jaipur": (26.9124, 75.7873),
    "Nagpur": (21.1458, 79.0882),
    "Pimpri-Chinchwad": (18.6298, 73.7997),
    "Pune": (18.5204, 73.8567)
}


# 2. Function
def get_city_data(lat, lon):
    data = rain.sel(LATITUDE=lat, LONGITUDE=lon, method='nearest').values
    return data[~np.isnan(data)]


# 3. MULTI-PLOT (ALL CITIES)
fig, axes = plt.subplots(4, 3, figsize=(16, 12))
axes = axes.flatten()

all_data = []
valid_idx = 0

for city, (lat, lon) in cities.items():
    data = get_city_data(lat, lon)

    if len(data) < 5:
        continue

    all_data.extend(data)

    mu, sigma = norm.fit(data)
    shape, loc, scale = gamma.fit(data, floc=0)

    x = np.linspace(min(data), max(data), 200)

    ax = axes[valid_idx]

    ax.hist(data, bins="auto", density=True, alpha=0.6)
    ax.plot(x, norm.pdf(x, mu, sigma), color='blue', linewidth=1.5)
    ax.plot(x, gamma.pdf(x, shape, loc, scale), color='orange', linewidth=1.5)

    ax.set_title(city, fontsize=10)
    ax.grid(alpha=0.3)

    valid_idx += 1

# Remove unused subplots
for j in range(valid_idx, len(axes)):
    fig.delaxes(axes[j])

# Global legend
handles = [
    plt.Line2D([0], [0], color='blue', linewidth=1.5),
    plt.Line2D([0], [0], color='orange', linewidth=1.5)
]
fig.legend(handles, ["Normal", "Gamma"], loc="upper right", fontsize=10)

plt.suptitle("Rainfall Distribution Across Cities", fontsize=18)
plt.tight_layout(rect=[0, 0, 0.95, 0.96])
plt.show()


# 4. COMBINED ANALYSIS
all_data = np.array(all_data)

mean = np.mean(all_data)
var = np.var(all_data)

print("\n=== Combined Statistics ===")
print("Mean:", mean)
print("Variance:", var)

# Fit distributions
mu, sigma = norm.fit(all_data)
shape, loc, scale = gamma.fit(all_data, floc=0)

# KS test
ks_norm = kstest(all_data, 'norm', args=(mu, sigma))
ks_gamma = kstest(all_data, 'gamma', args=(shape, loc, scale))

print("\nKS Normal:", ks_norm.statistic)
print("KS Gamma:", ks_gamma.statistic)
print("Better Fit:", "Gamma" if ks_gamma.statistic < ks_norm.statistic else "Normal")


# 5. DATA SMOOTHING
smoothed = np.convolve(all_data, np.ones(3)/3, mode='valid')

plt.figure(figsize=(8, 5))
plt.plot(all_data, label="Original")
plt.plot(smoothed, label="Smoothed")
plt.title("Data Smoothing (Moving Average)")
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# 6. TIME SERIES ANALYSIS
plt.figure(figsize=(8, 5))
plt.plot(all_data)
plt.title("Rainfall Time Series")
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.grid(alpha=0.3)
plt.show()


# 7. FINAL COMBINED PLOT
x = np.linspace(min(all_data), max(all_data), 300)

plt.figure(figsize=(8, 5))
plt.hist(all_data, bins="auto", density=True, alpha=0.6)

plt.plot(x, norm.pdf(x, mu, sigma), color='blue', linewidth=2, label="Normal")
plt.plot(x, gamma.pdf(x, shape, loc, scale), color='orange', linewidth=2, label="Gamma")

plt.title("Combined Rainfall Distribution (All Cities)", fontsize=14)
plt.xlabel("Rainfall")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Combined Statistics
# Mean: 1610.4142
# Variance: 3674356.2
# KS Normal: 0.35038598357887
# KS Gamma: 0.2874445623556556
# Better Fit: Gamma


# 8. HYPOTHESIS TESTING
from scipy.stats import ttest_ind

# Defining groups based on urbanization
high_urban = ["Ahmedabad", "Pune", "Pimpri-Chinchwad", "Bengaluru", "Hyderabad"]
low_urban = ["Meghalaya", "Bihar"]

high_data = []
low_data = []

for city, (lat, lon) in cities.items():
    data = get_city_data(lat, lon)

    if len(data) < 5:
        continue

    if city in high_urban:
        high_data.extend(data)
    elif city in low_urban:
        low_data.extend(data)

high_data = np.array(high_data)
low_data = np.array(low_data)

# 8.1 p-test (t-test + p-value)
print("\n8.1 p-test")
print("H0: Mean rainfall (High Urban) = Mean rainfall (Low Urban)")
print("H1: Mean rainfall (High Urban) ≠ Mean rainfall (Low Urban)")

t_stat, p_value = ttest_ind(high_data, low_data, equal_var=False)

print("\nTest Results:")
print("Sample size (High Urban):", len(high_data))
print("Sample size (Low Urban):", len(low_data))
print("Mean (High Urban):", np.mean(high_data))
print("Mean (Low Urban):", np.mean(low_data))
print("t-statistic:", t_stat)
print("p-value:", p_value)

# 8.2 Decision Rule
alpha = 0.05

print("\n8.2 Decision")
if p_value < alpha:
    print("Reject H0 → Significant difference in rainfall")
else:
    print("Fail to reject H0 → No significant difference detected")

print("Conclusion: Low urbanization regions show higher average rainfall")

# Test Results:
# Sample size (High Urban): 70
# Sample size (Low Urban): 28
# Mean (High Urban): 1384.7402
# Mean (Low Urban): 3315.8574
# t-statistic: -3.0262688173482957
# p-value: 0.0050167975987660905
# Reject H0 → Significant difference in rainfall


# 9. REGRESSION ANALYSIS
urban_index = {
    "Ahmedabad": 8, "Bengaluru": 9, "Pune": 9,
    "Pimpri-Chinchwad": 8, "Hyderabad": 9,
    "Meghalaya": 2, "Bihar": 3
}

X = []
Y = []

for city, (lat, lon) in cities.items():
    if city in urban_index:
        data = get_city_data(lat, lon)
        if len(data) > 0:
            X.append(urban_index[city])
            Y.append(np.mean(data))

X = np.array(X)
Y = np.array(Y)

coeffs = np.polyfit(X, Y, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X, Y)
plt.plot(X, np.polyval(coeffs, X))
plt.xlabel("Urbanization Index")
plt.ylabel("Mean Rainfall")
plt.title("Regression: Urbanization vs Rainfall")
plt.grid(alpha=0.3)
plt.show()

print("\nRegression Interpretation:")
if coeffs[0] < 0:
    print("Negative relationship: Higher urbanization → Lower rainfall")
else:
    print("Positive relationship: Higher urbanization → Higher rainfall")


# 10. CONFIDENCE INTERVALS

# 10.1. Analytical CI
from scipy.stats import t

n = len(all_data)
mean = np.mean(all_data)
std = np.std(all_data, ddof=1)

t_crit = t.ppf(0.975, df=n-1)
margin = t_crit * std / np.sqrt(n)

ci_lower = mean - margin
ci_upper = mean + margin

print("\n Analytical Confidence Interval ")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")


# 10.2. MONTE CARLO (BOOTSTRAP)
n_sim = 1000
sample_size = len(all_data)

means = []

for _ in range(n_sim):
    sample = np.random.choice(all_data, size=sample_size, replace=True)
    means.append(np.mean(sample))

means = np.array(means)

lower = np.percentile(means, 2.5)
upper = np.percentile(means, 97.5)

print("\n Monte Carlo Results ")
print(f"Mean estimate = {np.mean(means):.2f}")
print(f"95% Confidence Interval = [{lower:.2f}, {upper:.2f}]")

plt.figure(figsize=(8, 5))
plt.hist(means, bins=30, density=True, alpha=0.6)

plt.axvline(lower, color='red', linestyle='--', label='Lower CI')
plt.axvline(upper, color='green', linestyle='--', label='Upper CI')

plt.title("Monte Carlo Simulation (Bootstrap of Mean Rainfall)")
plt.xlabel("Mean Rainfall")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Monte Carlo Results
# Mean estimate = 1611.46
# 95% Confidence Interval = [1317.16, 1945.78]
