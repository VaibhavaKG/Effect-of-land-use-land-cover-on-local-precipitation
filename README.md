# Project: **Effect of Land Use on Local Precipitation 2005 - 2017**

> ## Objective:
Analyse how urbanisation (land use change) influences local rainfall patterns using statistical and data analysis methods. 

> ## Probability & Distributions
### Discrete vs. Continuous: 
Rainfall is continuous,and the discrete variables ("Drought Year," "Normal Year," "Flood Year").
### Probability Distribution Functions (PDFs):
To Test if our rainfall data follows a Gaussian (Normal) distribution or a Gamma distribution (very common for precipitation).
### Moments: 
Calculate the Mean, Variance, Skewness (is the rainfall skewed toward dry years?), and Kurtosis.
### Quantile Function: 
Determining the "1-in-10-year" extreme rainfall event using quantiles.

> ## Data Sources:
Rainfall: IMD gridded data \n
Land Use: WRIS, National Remote Sensing Centre (NRSC) only available for **2005-2017**

> ## Key Analysis
Comparing urban vs rural regions
Studying temporal changes (before vs after urban growth)
Analysing seasonal effects (monsoon vs non-monsoon)

> ## Correlation Methods
1. Linear Regression (OLS)
2. Multiple Linear Regression
3. Pearson Coefficient

> ## Clustering Methods 
K-Means Clustering (for Pattern Identification)
    * Cluster 1: High Rain, Low Urbanization (Wilderness/Forest).
    * Cluster 2: Low Rain, High Urbanization (Arid Cities).

> ## Expected Outcome
Detect correlation between urbanisation and rainfall
Identifying local-scale effects rather than global trends
Quantifying relationship strength using statistical metrics
