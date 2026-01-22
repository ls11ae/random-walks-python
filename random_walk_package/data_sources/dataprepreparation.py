import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs

def shark_data_filter(data_path):
    data = pd.read_csv(data_path)
    data = data.copy()
    data = data.sort_values("time")
    data["time"] = pd.to_datetime(data["time"]).dt.normalize()
    data["time"] = data["time"] + pd.Timedelta(hours=12)
    data = data.drop_duplicates(subset=["time"], keep="first")
    data = data.rename(columns={
        "ptt": "tag-local-identifier",
        "longitude": "location-long",
        "latitude": "location-lat",
        "time": "timestamp"
    })

    return data

def find_and_discard_outliers(data, lon_col= "location-long", lat_col= "location-lat"):
    coords = data[[lon_col, lat_col]].values
    #lof = LocalOutlierFactor(n_neighbors=20, contamination= 0.05)
    #outlier_labels = lof.fit_predict(coords)
    #outlier_mask = outlier_labels == -1
    center = np.median(coords, axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    
    outlier_mask = distances > (q3 + 2 * iqr)
    clean_data = coords[~outlier_mask]
    outliers = coords[outlier_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Original data with outliers marked
    ax1.scatter(coords[:, 0], coords[:, 1], 
                c='lightgray', s=30, alpha=0.5, 
                label=f'All points ({len(coords)})')
    
    ax1.scatter(outliers[:, 0], outliers[:, 1], 
                c='red', s=150, alpha=0.8, 
                label=f'Outliers ({len(outliers)})', 
                edgecolors='darkred', linewidth=2, 
                marker='X', zorder=5)
    
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title('Original Data with Outliers Marked', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cleaned data
    ax2.scatter(clean_data[:, 0], clean_data[:, 1], 
                c='skyblue', s=50, alpha=0.7, 
                label=f'Cleaned data ({len(clean_data)})',
                edgecolors='navy', linewidth=0.5)
    
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title('Cleaned Data (Outliers Removed)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Outliers detected:   {outlier_mask.sum()} ({100*outlier_mask.sum()/len(data):.2f}%)")
    print(f"Points retained:     {(~outlier_mask).sum()} ({100*(~outlier_mask).sum()/len(data):.2f}%)")
    
    cleaned_gdf = data[~outlier_mask].copy()
    
    return cleaned_gdf, outlier_mask

