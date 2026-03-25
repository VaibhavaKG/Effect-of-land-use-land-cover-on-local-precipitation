import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load dataset
ds = xr.open_dataset("yearly_sum.nc")
rain = ds["RAINFALL"]

# 2. Define locations
locations = {
    "Gurugram": {"lat": 28.46, "lon": 77.03},
    "Kochi": {"lat": 9.93, "lon": 76.26},
    "Pune": {"lat": 18.52, "lon": 73.85},
    "Chitta_RF": {"lat": 17.0, "lon": 78.5} }

# 3. Extract rainfall data
records = []

for name, coord in locations.items():
    ts = rain.sel(
        LATITUDE=coord["lat"],
        LONGITUDE=coord["lon"],
        method="nearest" )
    
    df = ts.to_dataframe().reset_index()
    df["Location"] = name
    records.append(df)

df = pd.concat(records)

# 4. Clean structure
df["YEAR"] = df["TIME"].dt.year
df = df[["YEAR", "Location", "RAINFALL"]]

print("\nData preview:\n", df.head())

# CORRELATION METHODS 
# Using YEAR as independent variable

print("\n Pearson Correlation (Rainfall vs Time)")
for loc in df["Location"].unique():
    sub = df[df["Location"] == loc]
    r, p = pearsonr(sub["YEAR"], sub["RAINFALL"])
    print(f"{loc}: r = {r:.3f}, p = {p:.3f}")

print("\n Spearman Correlation (Rainfall vs Time)")
for loc in df["Location"].unique():
    sub = df[df["Location"] == loc]
    r, p = spearmanr(sub["YEAR"], sub["RAINFALL"])
    print(f"{loc}: rho = {r:.3f}, p = {p:.3f}")

#  LINEAR REGRESSION 
print("\n Linear Regression (Rainfall ~ Year)")

model_time = smf.ols("RAINFALL ~ YEAR", data=df).fit()
print(model_time.summary())


#  MULTIPLE REGRESSION 
# Adding location effect
print("\n Multiple Regression (Rainfall ~ Year + Location)")

model_multi = smf.ols("RAINFALL ~ YEAR + C(Location)", data=df).fit()
print(model_multi.summary())


# TIME-LAGGED CORRELATION 

print("\n Time-Lagged Correlation (lag = 1 year)")

for loc in df["Location"].unique():
    sub = df[df["Location"] == loc].sort_values("YEAR")
    lag_corr = sub["RAINFALL"].corr(sub["RAINFALL"].shift(1))
    print(f"{loc}: lag-1 correlation = {lag_corr:.3f}")


#  T-TEST (Urban vs Forest proxy) 

urban = df[df["Location"] != "Chitta_RF"]["RAINFALL"]
forest = df[df["Location"] == "Chitta_RF"]["RAINFALL"]

t_stat, p_val = ttest_ind(urban, forest)

print("\n--- T-Test (Urban locations vs Forest) ---")
print(f"t = {t_stat:.3f}, p = {p_val:.3f}")

#  VISUALIZATION 
sns.set_style("whitegrid")

# Time series plot
plt.figure()
sns.lineplot(data=df, x="YEAR", y="RAINFALL", hue="Location", marker="o")
plt.title("Year-wise Rainfall Variation")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.legend(title="Location")
plt.show()

# Scatter with trend
plt.figure()
sns.scatterplot(data=df, x="YEAR", y="RAINFALL", hue="Location")
sns.regplot(data=df, x="YEAR", y="RAINFALL", scatter=False)
plt.title("Rainfall Trend with Time")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(data=df, x="Location", y="RAINFALL")
plt.title("Rainfall Distribution Across Locations")
plt.xlabel("Location")
plt.ylabel("Rainfall (mm)")
plt.show()

# Violin plot
plt.figure()
sns.violinplot(data=df, x="Location", y="RAINFALL")
plt.title("Rainfall Distribution Shape")
plt.xlabel("Location")
plt.ylabel("Rainfall (mm)")
plt.show()

# Histogram
plt.figure()
sns.histplot(data=df, x="RAINFALL", hue="Location", kde=True)
plt.title("Rainfall Frequency Distribution")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Count")
plt.show()
