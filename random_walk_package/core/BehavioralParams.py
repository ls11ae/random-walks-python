from random_walk_package.data_sources.geo_fetcher import lonlat_bbox_to_utm, utm_bbox_to_lonlat, fetch_landcover_data
import copernicusmarine #as for now I already got the data forced to be connected to my shark datasets
import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.stats import circmean
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt 

def shark_data_filter(data_path):
    data = pd.read_csv(data_path)
    data = data.copy()
    data = data.sort_values("time")
    data["time"] = pd.to_datetime(data["time"]).dt.normalize()
    data["time"] = data["time"] + pd.Timedelta(hours=12)
    data = data.drop_duplicates(subset=["time"], keep="first")
    max_lat = data["latitude"].max()
    max_long = data["longitude"].max()
    min_lat =data["latitude"].min()
    min_long = data["longitude"].min()
    bbox = lonlat_bbox_to_utm(min_long, min_lat, max_long, max_lat, "4326")
   
    return data, bbox


def interpolate_to_regular_timesteps(data, time_step):
        """
        Interpolate sparse tracking data to regular time intervals.
    
        """
        print(f"Interpolating data to regular {time_step} intervals...")
        print(f"Original data points: {len(data)}")
        
        # Create regular time grid
        time_range = pd.date_range(
            start=data['time'].min(),
            end=data['time'].max(),
            freq=time_step
        )
        
        # Set date as index for interpolation
        df_indexed = data.set_index('time')
        
        # Reindex to regular grid and interpolate
        df_regular = df_indexed.reindex(time_range)
        
        df_regular['longitude'] = df_regular['longitude'].interpolate(method='linear')
        df_regular['latitude'] = df_regular['latitude'].interpolate(method='linear')
        

        df_regular = df_regular.dropna()
        
        # Reset index and rename
        df_regular = df_regular.reset_index()
        df_regular = df_regular.rename(columns={'index': 'time'})
        
        data = df_regular
        
        print(f"Interpolated data points: {len(data)}")
        print(f"Time step: {time_step}")
        
        # Calculate actual time differences to verify
        time_diffs = np.diff(data['time']).astype('timedelta64[s]').astype(float)
        print(f"Mean time interval: {np.mean(time_diffs)/3600:.2f} hours")
        print(f"Std time interval: {np.std(time_diffs)/3600:.2f} hours")
        
landcover_classes = {
    10: "LAND", 
    90: "OCEAN"
}
""" Estimate shark movement parameters from tracking data: 
Movement parameters including:
            - step_lengths: array of step sizes (m)
            - bearings: array of turning angles (radians)
            - velocities: array of speeds (m/s)
            - time_intervals: array of time differences (seconds)
            - mean_velocity: mean swimming speed
            - diffusivity: movement diffusivity
            - directional_bias: mean direction preference
            - turn_concentration: turning angle concentration (kappa)
"""
def coordinates_to_xy(data):
        """Transforms lat/lon to azimuthal equidistant projection."""
        x, y = Proj(data['longitude'].values, 
                         data['latitude'].values)
        data['x'] = x
        data['y'] = y
        
        
def step_length(): 
    dx = np.diff(data['x'].values)
    dy = np.diff(data['y'].values)
    step_lengths = np.sqrt(dx**2 + dy**2)
    return dx, dy, step_lengths

def turning_angles( dx,dy) :
    """
    Calculates turning angles = change in direction between steps (radians, -π to π), where 
    positive = left turn, negative = right turn;
    and bearings_abs = mean preferred direction in radians .
    """
    bearings_abs = np.arctan2(dy, dx)
    turning_angles = np.diff(bearings_abs)
    turning_angles = np.arctan2(np.sin(turning_angles), np.cos(turning_angles))   
    directional_bias = circmean(bearings_abs)
    return bearings_abs, turning_angles

def time_intervals(data): 
    dates = pd.to_datetime(data['time'])
    time_diffs = np.diff(dates).astype('timedelta64[s]').astype(float)
    return time_diffs

def behavioural_speed(step_length, time_diffs, use_lit_prior, prior_weight=0.5,
                      literature_cruising_min_ms = 0.67, literature_cruising_max_ms = 1.34):
    """
    Calculate mean behavioral velocity with optional literature prior. Calculate
    velocities bas                                                                                                                          ed on some predefined step_length and time-diffs. To avoid unrealistic estimate, can be optionally validated
    or meaned by literature_cruising_min_ms and literature_cruising_max_ms
    """
    velocities = step_length / time_diffs
    mean_velocity = np.median(velocities[velocities > 0])
    
    mean_velocity_literature = (literature_cruising_min_ms + literature_cruising_max_ms) / 2  # 1.005 m/s ≈ 3.6 km/h
        
    if use_lit_prior:
            # Weighted average between observed and literature
        mean_velocity = (1 - prior_weight) * mean_velocity + \
                           prior_weight * mean_velocity_literature
    return mean_velocity
    
def directional_persistance(self, angles):
        """Estimate von Mises concentration parameter using MLE."""
        r = np.abs(np.mean(np.exp(1j * angles)))
        if r < 0.53:
            kappa = 2 * r + r**3 + 5 * r**5 / 6
        elif r < 0.85:
            kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            kappa = 1 / (r**3 - 4 * r**2 + 3 * r)
        return kappa 
    
def compare_with_reference_speeds(velocities, mean_velocity):
        """
        Compare calculated velocities with published tiger shark movement rates.
        
        Based on telemetry studies
        
        """

        print("COMPARISON WITH https://www.oceanactionhub.org/tiger-shark-speed/")
        
        
        # Reference values from literature
        ref_cruising_min_ms = 0.67  # m/s
        ref_cruising_max_ms = 1.34  # m/s
        ref_cruising_min_kmh = 2.4  # km/h
        ref_cruising_max_kmh = 4.8  # km/h
        
        ref_directed_min_ms = 0.56
        ref_directed_max_ms = 0.83
        ref_directed_min_kmh = 2.0
        ref_directed_max_kmh = 3.0
        
        # Calculate statistics
        mean_kmh = mean_velocity * 3.6
        min_kmh = np.min(velocities) * 3.6
        max_kmh = np.max(velocities) * 3.6
        
        print(f"\nYour Data:")
        print(f"  Mean velocity:     {mean_velocity:.3f} m/s  =  {mean_kmh:.2f} km/h")
        print(f"  Min velocity:      {np.min(velocities):.3f} m/s  =  {min_kmh:.2f} km/h")
        print(f"  Max velocity:      {np.max(velocities):.3f} m/s  =  {max_kmh:.2f} km/h")
        
        # Validation checks
        print(f"\nValidation:")
        
        # Check if mean is within cruising range
        if ref_cruising_min_ms <= mean_velocity <= ref_cruising_max_ms:
            print(f"Mean velocity is within routine cruising range")
            status_mean = "REALISTIC"
        else:
            print(f"  ⚠ Mean velocity is HIGHER than typical cruising (check for outliers)")
            status_mean = "TOO HIGH"
    
        # Check for unrealistic outliers
        outliers_high = np.sum(velocities > 5.0)  # > 18 km/h is very fast
        outliers_low = np.sum(velocities < 0.1)   # < 0.36 km/h is very slow
        
        if outliers_high > 0:
            print(f"  ⚠ Found {outliers_high} measurements > 5 m/s (18 km/h) - may be unrealistic")
        if outliers_low > 0:
            print(f"  ⚠ Found {outliers_low} measurements < 0.1 m/s (0.36 km/h) - near stationary")
        
        return {
            'mean_velocity_ms': mean_velocity,
            'mean_velocity_kmh': mean_kmh,
            'status': status_mean,
        }

def diffusivity(step_lengths, time_diffs):
    """
    Calculate movement diffusivity (spread parameter).
    
    diffusivity  = Diffusion coefficient (m²/s);
    Higher = more random/spread out movement
    """
    mean_squared_displacement = np.mean(step_lengths**2)
    mean_time_step = np.median(time_diffs)
    diffusivity = mean_squared_displacement / (4 * mean_time_step)
    return diffusivity


data_path = "/home/poiosh/movement_py/shark_13_with_currents.csv"
data, bbox = shark_data_filter(data_path)
fetch_landcover_data(bbox)