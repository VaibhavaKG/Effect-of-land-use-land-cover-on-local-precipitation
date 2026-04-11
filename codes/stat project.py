import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
geojson_folder = r"C:\Users\srush\Downloads\SHAPEFILES\states"
output_folder = r"C:\Users\srush\Downloads\stat_project_results"
os.makedirs(output_folder, exist_ok=True)
years = range(2005, 2018)
results = []
def process_year(year, geo_file, gdf_buffered):
    year2 = year + 1
    year3 = f"{year%100:02d}{year2%100:02d}"
    path = rf"C:\Users\srush\Downloads\lulc\LULC 250k {year}-{year2}\LULC 250k {year3}\lulc250k_{year3}.tif"
    try:
        with rasterio.open(path) as src:
            gdf_proj = gdf_buffered.to_crs(src.crs)
            clipped, _ = mask(src, gdf_proj.geometry, crop=True)
            raster = clipped[0]
            builtup = (raster == 1)
            builtup = np.logical_and(builtup, raster != src.nodata)
            pixel_area = abs(src.res[0] * src.res[1])
            builtup_area_km2 = np.sum(builtup) * pixel_area / 1e6
            return {
                "region": geo_file,
                "year": year,
                "builtup_area": builtup_area_km2}
    except Exception as e:
        print(f"Error for {geo_file} {year}: {e}")
        return None
for geo_file in os.listdir(geojson_folder):
    if not geo_file.endswith(".geojson"):
        continue
    geo_path = os.path.join(geojson_folder, geo_file)
    gdf = gpd.read_file(geo_path)
    outputs = Parallel(n_jobs=-1)(
        delayed(process_year)(year, geo_file, gdf)
        for year in years
    )
    for out in outputs:
        if out is not None:
            results.append(out)
print(results)
df = pd.DataFrame(results)
df = df.sort_values(["region", "year"])
'df.to_csv(os.path.join(output_folder," builtup_area_states.csv"), index=False)'
#cities
geojson_folder = r"C:\Users\srush\Downloads\SHAPEFILES" 
for geo_file in os.listdir(geojson_folder):
    if not geo_file.endswith(".geojson"):
        continue
    geo_path = os.path.join(geojson_folder, geo_file)
    gdf = gpd.read_file(geo_path)
    gdf_meter = gdf.to_crs("EPSG:3857")
    gdf_buffered = gdf_meter.buffer(25000)  # 25 km buffer
    gdf_buffered = gpd.GeoDataFrame(geometry=gdf_buffered, crs="EPSG:3857")
    outputs = Parallel(n_jobs=-1)(
        delayed(process_year)(year, geo_file, gdf)
        for year in years)
    for out in outputs:
        if out is not None:
            results.append(out)
print(results)
df = pd.DataFrame(results)
df = df.sort_values(["region", "year"])
'df.to_csv(os.path.join(output_folder," builtup_area_cities.csv"), index=False) '       
#%%
import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import re
geojson_folder = r"C:\Users\srush\Downloads\SHAPEFILES\states"
nc_folder = r"C:\Users\srush\Downloads\rainfall"
results = []
year_files = defaultdict(list)
for file in os.listdir(nc_folder):
    if file.endswith(".nc"):
        match = re.search(r"\d{4}", file) 
        if match:
            year = match.group()
            year_files[year].append(os.path.join(nc_folder, file))
for geo_file in os.listdir(geojson_folder):
    if not geo_file.endswith(".geojson"):
        continue
    geo_path = os.path.join(geojson_folder, geo_file)
    gdf = gpd.read_file(geo_path)
    for year in sorted(year_files.keys()):
        files = year_files[year]
        try:
            ds = xr.open_mfdataset(files, combine='by_coords')
            rain = ds['RAINFALL']
            rain = rain.sum(dim='TIME')
            rain = rain.rename({
                "LONGITUDE": "lon",
                "LATITUDE": "lat"})
            rain = rain.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            gdf = gdf.to_crs("EPSG:3857")   
            rain = rain.rio.write_crs("EPSG:4326")
            if rain.lat[0] > rain.lat[-1]:
                rain = rain.sortby("lat")
            if rain.lon.max() > 180:
                rain = rain.assign_coords(lon=((rain.lon + 180) % 360) - 180).sortby("lon")
            gdf_proj = gdf.to_crs(rain.rio.crs)
            clipped = rain.rio.clip(gdf_proj.geometry, gdf_proj.crs, drop=True)
            data = clipped.values
            valid = data[~np.isnan(data)]
            mean_rain = np.nan if len(valid) == 0 else np.mean(valid)
            results.append({
                "region":geo_file,
                "year": int(year),
                "mean_rainfall": mean_rain})
            print(f"{geo_file}-{year}:{mean_rain:.2f}")
        except Exception as e:
            print(f"Skipping {year}: {e}")
df = pd.DataFrame(results)
df = df.sort_values(["region", "year"])
#df.to_csv(os.path.join(r"C:\Users\srush\Downloads\stat_project_results","yearly_mean_rainfall_states.csv"), index=False)
#%%
import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import re
geojson_folder = r"C:\Users\srush\Downloads\SHAPEFILES\states"
nc_folder = r"C:\Users\srush\Downloads\rainfall"
results = []
year_files = defaultdict(list)
for file in os.listdir(nc_folder):
    if file.endswith(".nc"):
        match = re.search(r"\d{4}", file)  
        if match:
            year = match.group()
            year_files[year].append(os.path.join(nc_folder, file))
for geo_file in os.listdir(geojson_folder):
    if not geo_file.endswith(".geojson"):
        continue
    geo_path = os.path.join(geojson_folder, geo_file)
    gdf = gpd.read_file(geo_path)
    for year in sorted(year_files.keys()):
        files = year_files[year]
        try:
            ds = xr.open_mfdataset(files, combine='by_coords')
            rain_jjas = ds['RAINFALL'].sel(TIME=ds.TIME.dt.month.isin([6, 7, 8, 9]))            
            rain = rain_jjas.sum(dim='TIME')
            rain = rain.rename({
                "LONGITUDE": "lon",
                "LATITUDE": "lat"})
            rain = rain.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            gdf = gdf.to_crs("EPSG:3857") 
            buffered = gdf.geometry.buffer(25000)  # 25 km
            gdf = gdf.set_geometry(buffered)
            gdf = gdf.to_crs("EPSG:4326") 
            rain = rain.rio.write_crs("EPSG:4326")
            if rain.lat[0] > rain.lat[-1]:
                rain = rain.sortby("lat")
            if rain.lon.max() > 180:
                rain = rain.assign_coords(lon=((rain.lon + 180) % 360) - 180).sortby("lon")
            gdf_proj = gdf.to_crs(rain.rio.crs)
            clipped = rain.rio.clip(gdf_proj.geometry, gdf_proj.crs, drop=True)
            data = clipped.values
            print("Min/Max:", np.nanmin(data), np.nanmax(data))
            valid = data[~np.isnan(data)]
            mean_rain = np.nan if len(valid) == 0 else np.mean(valid)
            results.append({"region":geo_file,
                "year": int(year),
                "mean_rainfall": mean_rain })
            print(f"{year}:{mean_rain:.2f}")
        except Exception as e:
            print(f"Skipping {year}: {e}")
df = pd.DataFrame(results)
df = df.sort_values(["region", "year"])
#df.to_csv(os.path.join(r"C:\Users\srush\Downloads\stat_project_results","jjas_mean_rain_states.csv"), index=False)
#%%
## state:yearly

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
df = pd.read_csv("C:/Users/srush/Downloads/stat_project_results/yearly_mean_rainfall_builtup_states.csv")
correlations = []
for region in df['region'].unique():
    subset = df[df['region'] == region].dropna(subset=['builtup_area', 'mean_rainfall'])
    if len(subset) > 1:
        corr_val = subset['builtup_area'].corr(subset['mean_rainfall'])
        correlations.append({'region': region, 'correlation': corr_val})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

#PLOT 1: Bar Chart
plt.figure(figsize=(14, 10))
sns.barplot(data=corr_df, x='correlation', y='region', palette='coolwarm')
plt.title('Correlation: Rainfall vs Built-up Area by Region', fontsize=36, pad=25,fontweight='bold')
plt.xlabel('Pearson Correlation Coefficient', fontsize=30,fontweight='bold')
plt.ylabel('Region', fontsize=30,fontweight='bold')
plt.xticks(fontsize=25,fontweight='bold')
plt.yticks(fontsize=25,fontweight='bold')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "pcorrelation_states.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

#  PLOT 2: Faceted Scatter Plots with P-Values 
def annotate_regression(x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if len(x[mask]) > 1:
        slope, intercept, r_val, p_val, std_err = linregress(x[mask], y[mask])
        ax = plt.gca()
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        annotation = f"r = {r_val:.2f}\n{p_text}"        
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                fontsize=30,fontweight='bold', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
g = sns.FacetGrid(df, col="region", col_wrap=4, sharex=False, sharey=False, height=5, aspect=1.2)
g.map(sns.regplot, "builtup_area", "mean_rainfall", 
      scatter_kws={'alpha':0.5, 's':60}, line_kws={'color':'red'})
g.map(annotate_regression, "builtup_area", "mean_rainfall")
g.set_titles(col_template="{col_name}", size=30,fontweight='bold')
g.set_axis_labels("Built-up Area", "Mean Rainfall", fontsize=30,fontweight='bold')
g.fig.subplots_adjust(wspace=0.5, hspace=0.5) 
for ax in g.axes.flatten():
            ax.tick_params(labelsize=30)  
for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')  
g.fig.suptitle('Built-up Area vs Yearly Rainfall (states)', y=1.05, fontsize=35,fontweight='bold')
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "correlation_states.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
#%%
#state: jjas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. Load Data
df = pd.read_csv("C:/Users/srush/Downloads/stat_project_results/jjas_mean_rain_states.csv")

# 2. Calculate correlations for the Bar Chart
correlations = []
for region in df['region'].unique():
    subset = df[df['region'] == region].dropna(subset=['builtup_area', 'mean_rainfall'])
    if len(subset) > 1:
        corr_val = subset['builtup_area'].corr(subset['mean_rainfall'])
        correlations.append({'region': region, 'correlation': corr_val})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

#PLOT 1: Bar Chart
plt.figure(figsize=(14, 10))
sns.barplot(data=corr_df, x='correlation', y='region', palette='coolwarm')
plt.title('Correlation: JJAS Rainfall vs Built-up Area by Region', fontsize=36, pad=25,fontweight='bold')
plt.xlabel('Pearson Correlation Coefficient', fontsize=30,fontweight='bold')
plt.ylabel('Region', fontsize=30,fontweight='bold')
plt.xticks(fontsize=25,fontweight='bold')
plt.yticks(fontsize=25,fontweight='bold')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "jjas_pcorrelation_states.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

#  PLOT 2: Faceted Scatter Plots with P-Values 
def annotate_regression(x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if len(x[mask]) > 1:
        slope, intercept, r_val, p_val, std_err = linregress(x[mask], y[mask])
        ax = plt.gca()
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        annotation = f"r = {r_val:.2f}\n{p_text}"        
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                fontsize=30,fontweight='bold', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
g = sns.FacetGrid(df, col="region", col_wrap=4, sharex=False, sharey=False, height=5, aspect=1.2)
g.map(sns.regplot, "builtup_area", "mean_rainfall", 
      scatter_kws={'alpha':0.5, 's':60}, line_kws={'color':'red'})
g.map(annotate_regression, "builtup_area", "mean_rainfall")
g.set_titles(col_template="{col_name}", size=30,fontweight='bold')
g.set_axis_labels("Built-up Area", "Mean Rainfall", fontsize=30,fontweight='bold')
g.fig.subplots_adjust(wspace=0.5, hspace=0.5) 
for ax in g.axes.flatten():
            ax.tick_params(labelsize=30)  
for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')  
g.fig.suptitle('Built-up Area vs JJAS Rainfall (States)', y=1.05, fontsize=35,fontweight='bold')
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "jjas_correlation_states.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

#%%
# city: yearly
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
df = pd.read_csv("C:/Users/srush/Downloads/stat_project_results/yearly_mean_rainfall_builtup_cities.csv")
correlations = []
for region in df['region'].unique():
    subset = df[df['region'] == region].dropna(subset=['builtup_area', 'mean_rainfall'])
    if len(subset) > 1:
        corr_val = subset['builtup_area'].corr(subset['mean_rainfall'])
        correlations.append({'region': region, 'correlation': corr_val})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

#PLOT 1: Bar Chart
plt.figure(figsize=(14, 10))
sns.barplot(data=corr_df, x='correlation', y='region', palette='coolwarm')
plt.title('Correlation: Rainfall vs Built-up Area (cities)', fontsize=36, pad=25,fontweight='bold')
plt.xlabel('Pearson Correlation Coefficient', fontsize=30,fontweight='bold')
plt.ylabel('Region', fontsize=30,fontweight='bold')
plt.xticks(fontsize=25,fontweight='bold')
plt.yticks(fontsize=25,fontweight='bold')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "pcorrelation_cities.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

#  PLOT 2: Faceted Scatter Plots with P-Values 
def annotate_regression(x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if len(x[mask]) > 1:
        slope, intercept, r_val, p_val, std_err = linregress(x[mask], y[mask])
        ax = plt.gca()
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        annotation = f"r = {r_val:.2f}\n{p_text}"        
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                fontsize=30,fontweight='bold', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
g = sns.FacetGrid(df, col="region", col_wrap=4, sharex=False, sharey=False, height=5, aspect=1.2)
g.map(sns.regplot, "builtup_area", "mean_rainfall", 
      scatter_kws={'alpha':0.5, 's':60}, line_kws={'color':'red'})
g.map(annotate_regression, "builtup_area", "mean_rainfall")
g.set_titles(col_template="{col_name}", size=30,fontweight='bold')
g.set_axis_labels("Built-up Area", "Mean Rainfall", fontsize=30,fontweight='bold')
g.fig.subplots_adjust(wspace=0.5, hspace=0.5) 
for ax in g.axes.flatten():
            ax.tick_params(labelsize=30)  
for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')  
g.fig.suptitle('Built-up Area vs Yearly Rainfall (cities)', y=1.05, fontsize=35,fontweight='bold')
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "correlation_cities.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
#%%
#city: jjas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. Load Data
df = pd.read_csv("C:/Users/srush/Downloads/stat_project_results/jjas_mean_rain_cities.csv")

# 2. Calculate correlations for the Bar Chart
correlations = []
for region in df['region'].unique():
    subset = df[df['region'] == region].dropna(subset=['builtup_area', 'mean_rainfall'])
    if len(subset) > 1:
        corr_val = subset['builtup_area'].corr(subset['mean_rainfall'])
        correlations.append({'region': region, 'correlation': corr_val})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

#PLOT 1: Bar Chart
plt.figure(figsize=(14, 10))
sns.barplot(data=corr_df, x='correlation', y='region', palette='coolwarm')
plt.title('Correlation: JJAS Rainfall vs Built-up Area (cities)', fontsize=36, pad=25,fontweight='bold')
plt.xlabel('Pearson Correlation Coefficient', fontsize=30,fontweight='bold')
plt.ylabel('Region', fontsize=30,fontweight='bold')
plt.xticks(fontsize=25,fontweight='bold')
plt.yticks(fontsize=25,fontweight='bold')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "jjas_pcorrelation_cities.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

#  PLOT 2: Faceted Scatter Plots with P-Values 
def annotate_regression(x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if len(x[mask]) > 1:
        slope, intercept, r_val, p_val, std_err = linregress(x[mask], y[mask])
        ax = plt.gca()
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        annotation = f"r = {r_val:.2f}\n{p_text}"        
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                fontsize=30,fontweight='bold', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
g = sns.FacetGrid(df, col="region", col_wrap=4, sharex=False, sharey=False, height=5, aspect=1.2)
g.map(sns.regplot, "builtup_area", "mean_rainfall", 
      scatter_kws={'alpha':0.5, 's':60}, line_kws={'color':'red'})
g.map(annotate_regression, "builtup_area", "mean_rainfall")
g.set_titles(col_template="{col_name}", size=30,fontweight='bold')
g.set_axis_labels("Built-up Area", "Mean Rainfall", fontsize=30,fontweight='bold')
g.fig.subplots_adjust(wspace=0.5, hspace=0.5) 
for ax in g.axes.flatten():
            ax.tick_params(labelsize=30)  
for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')  
g.fig.suptitle('Built-up Area vs JJAS Rainfall (cities)', y=1.05, fontsize=35,fontweight='bold')
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "jjas_correlation_cities.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
#%%
#INDIA MOMENTS
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
nc_file = "C:/Users/srush/Downloads/rainfall/New folder/yearly_sum.nc"
ds = xr.open_dataset(nc_file)
if "LONGITUDE" in ds.coords:
    ds = ds.rename({"LONGITUDE": "lon", "LATITUDE": "lat"})
jjas_data = ds['RAINFALL'].sel(TIME=ds.TIME.dt.month.isin([6, 7, 8, 9]))
yearly_data=ds['RAINFALL']
def  moments(rain_data):
    spatial_mean = rain_data.mean(dim='TIME')
    spatial_var = rain_data.var(dim='TIME')
    diff = rain_data - spatial_mean
    spatial_skew = (diff**3).mean(dim='TIME') /(spatial_var**1.5)
    spatial_kurt = (diff**4).mean(dim='TIME') /(spatial_var**2) - 3
    fig, axes = plt.subplots(2, 2, figsize=(22, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    moments_data = [spatial_mean, spatial_var, spatial_skew, spatial_kurt]
    titles = ['Mean Rainfall (mm)', 'Rainfall Variance', 'Rainfall Skewness', 'Excess Kurtosis']
    cmaps = ['YlGnBu', 'YlOrRd', 'RdBu_r', 'magma']
    for i, ax in enumerate(axes):
        im = moments_data[i].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cmap=cmaps[i], 
        add_colorbar=True,
        robust=True,
        cbar_kwargs={'shrink': 0.7, 'label': ''})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_title(titles[i], fontsize=30,fontweight='bold',pad=15)
        im.colorbar.ax.tick_params(labelsize=30)
        im.colorbar.set_label(titles[i], fontsize=30, labelpad=15)
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.xlabel_style = {'size': 30}
        gl.ylabel_style = {'size': 30}
        gl.top_labels = False
        gl.right_labels = False
moments(jjas_data)
plt.suptitle("Spatial Statistical Moments of JJAS Rainfall (2005-2017)", fontsize=35, fontweight='bold', y=1.02)
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "jjas_moments_india.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
moments(yearly_data)
plt.suptitle("Spatial Statistical Moments of yearly Rainfall (2005-2017)", fontsize=35,fontweight='bold',y=1.02)
plt.tight_layout()
save_path = os.path.join(r"C:\Users\srush\Downloads\stat_project_results", "yearly_moments_india.jpg") 
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

#%% 
#MOMENTS of mean rainfall

import pandas as pd
import numpy as np

def moments(file):
    moments_df = file.groupby('region')['mean_rainfall'].agg([
        ('Mean', 'mean'),
        ('Variance', 'var'),
        ('Skewness', lambda x: x.skew()),
        ('Kurtosis', lambda x: x.kurt())
        ]).reset_index()
    moments_df = moments_df.round(3)
    print(moments_df.head())
df = pd.read_csv("C:/Users/srush/Downloads/stat_project_results/yearly_mean_rainfall_builtup_states.csv")
print("moments of states yearly rainfall:")
print("\n")
moments(df)
print("\n")
df= pd.read_csv("C:/Users/srush/Downloads/stat_project_results/yearly_mean_rainfall_builtup_cities.csv")
print("moments of cities yearly rainfall:")
print("\n")
moments(df)
print("\n")
df = pd.read_csv("C:/Users/srush/Downloads/stat_project_results/jjas_mean_rain_states.csv")
print("moments of states jjas rainfall")
print("\n")
moments(df)
print("\n")
df= pd.read_csv("C:/Users/srush/Downloads/stat_project_results/jjas_mean_rain_cities.csv")
print("\n")
print("moments of cities yearly rainfall")
print("\n")
moments(df)

#%%
#moments per region
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray
import os
from scipy.stats import linregress
nc_file = r"C:\Users\srush\Downloads\rainfall\New folder\yearly_sum.nc"
geojson_folder = r"C:\Users\srush\Downloads\SHAPEFILES\states"
output_folder = r"C:\Users\srush\Downloads\stat_project_results\Regional_Zoomed_Plots"
geojson_folder1 = r"C:\Users\srush\Downloads\SHAPEFILES"
os.makedirs(output_folder, exist_ok=True)
ds = xr.open_dataset(nc_file)
if "LONGITUDE" in ds.coords:
    ds = ds.rename({"LONGITUDE": "lon", "LATITUDE": "lat"})
jjas = ds['RAINFALL'].sel(TIME=ds.TIME.dt.month.isin([6, 7, 8, 9])).rio.write_crs("EPSG:4326")
rainfall=ds['RAINFALL']
def spatial_moments(jjas, geojson_folder):
    s_mean = jjas.mean(dim='TIME')
    s_var  = jjas.var(dim='TIME')
    diff   = jjas - s_mean
    s_skew = (diff**3).mean(dim='TIME') / (s_var**1.5)
    s_kurt = (diff**4).mean(dim='TIME') / (s_var**2) - 3
    time = jjas.TIME.dt.year.values
    def calc_trend(data):
        if np.all(np.isnan(data)):
            return np.nan
        slope, _, _, _, _ = linregress(time, data)
        return slope
    trend = xr.apply_ufunc(
        calc_trend, jjas,
        input_core_dims=[['TIME']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    moments_list = [s_mean, s_var, s_skew, s_kurt, trend]
    titles = ['Mean Rain (mm)', 'Variance ', 'Skewness', 'Kurtosis', 'Trend (mm/yr)']
    cmaps = ['YlGnBu', 'YlOrRd', 'RdBu_r', 'magma', 'RdBu_r']
    for file in os.listdir(geojson_folder):
        if file.endswith(".geojson"):
            region_name = file.replace(".geojson", "")
            print(f"Creating plots for: {region_name}")
            gdf_orig = gpd.read_file(
                os.path.join(geojson_folder, file)
            ).to_crs("EPSG:4326")
            minx, miny, maxx, maxy = gdf_orig.total_bounds
            pad_x = (maxx - minx) * 0.25
            pad_y = (maxy - miny) * 0.25
            extent = [minx - pad_x, maxx + pad_x,
                      miny - pad_y, maxy + pad_y]
            fig, axes = plt.subplots(
                2, 3, figsize=(24, 14),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            axes = axes.flatten()
            gdf_buffer = gdf_orig.to_crs("EPSG:3857")
            gdf_buffer["geometry"] = gdf_buffer.buffer(25000)
            gdf_buffer = gdf_buffer.to_crs("EPSG:4326")
            for i in range(len(moments_list)):
                ax = axes[i]
                try:
                    clipped = moments_list[i].rio.clip(
                        gdf_buffer.geometry,
                        gdf_buffer.crs,
                        drop=True)
                    im = clipped.plot(
                        ax=ax,
                        transform=ccrs.PlateCarree(),
                        cmap=cmaps[i],
                        add_colorbar=False,
                        robust=True)
                    cbar = plt.colorbar(
                        im, ax=ax,
                        orientation='vertical',
                        shrink=0.7,
                        pad=0.03,
                        aspect=20)
                    cbar.ax.tick_params(labelsize=25)
                    cbar.set_label(titles[i], fontsize=25,fontweight='bold')
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    gdf_orig.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2)
                    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                    ax.set_title(titles[i], fontsize=25, fontweight='bold')
                    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {'size': 25}
                    gl.ylabel_style = {'size': 25}
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}",
                            ha='center', transform=ax.transAxes)
            for j in range(len(moments_list), len(axes)):
                axes[j].remove()
            plt.suptitle(
                f"{region_name} - JJAS Seasonal Statistics",
                fontsize=35, fontweight='bold', y=0.97)
            plt.subplots_adjust(
                left=0.05, right=0.92,
                top=0.90, bottom=0.05,
                wspace=0.5, hspace=0.30 )
            save_path = os.path.join(
                output_folder,
                f"{region_name}_ZOOMED.jpg")
            plt.savefig(save_path, dpi=250, bbox_inches='tight')
            plt.show()
            plt.close()
spatial_moments(jjas,geojson_folder)                
spatial_moments(jjas,geojson_folder1)                

