import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, kstest


ds = xr.open_dataset("yearly_sum.nc")
rain = ds['RAINFALL']


# City coordinates
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


# Function
def get_city_data(lat, lon):
    data = rain.sel(LATITUDE=lat, LONGITUDE=lon, method='nearest').values
    return data[~np.isnan(data)]


# MULTI-PLOT (ALL CITIES)
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
    ax.plot(x, norm.pdf(x, mu, sigma), linewidth=1.5)
    ax.plot(x, gamma.pdf(x, shape, loc, scale), linewidth=1.5)

    ax.set_title(city, fontsize=10)
    ax.grid(alpha=0.3)

    valid_idx += 1

# Remove unused subplots
for j in range(valid_idx, len(axes)):
    fig.delaxes(axes[j])

# Global legend
handles = [
    plt.Line2D([0], [0], linewidth=1.5),
    plt.Line2D([0], [0], linewidth=1.5)
]
fig.legend(handles, ["Normal", "Gamma"], loc="upper right", fontsize=10)

plt.suptitle("Rainfall Distribution Across Cities", fontsize=18)
plt.tight_layout(rect=[0, 0, 0.95, 0.96])
plt.show()


# COMBINED ANALYSIS
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


# FINAL COMBINED PLOT
x = np.linspace(min(all_data), max(all_data), 300)

plt.figure(figsize=(8, 5))
plt.hist(all_data, bins="auto", density=True, alpha=0.6)

plt.plot(x, norm.pdf(x, mu, sigma), linewidth=2, label="Normal")
plt.plot(x, gamma.pdf(x, shape, loc, scale), linewidth=2, label="Gamma")

plt.title("Combined Rainfall Distribution (All Cities)", fontsize=14)
plt.xlabel("Rainfall")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# Combined Statistics
#Mean: 1610.4142
#Variance: 3674356.2
#KS Normal: 0.35038598357887
#KS Gamma: 0.2874445623556556
#Better Fit: Gamma
