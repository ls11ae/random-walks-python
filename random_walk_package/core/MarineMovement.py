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
class MarineMovement:
    def __init__(self, age_class="adult"):
        self.age_class = age_class.lower()

        # Projection: azimuthal equidistant
        self.proj = Proj(proj="aeqd", lat_0=0, lon_0=0)

        # Results stored
        self.data = None
        self.dx = None
        self.dy = None
        self.step_lengths = None
        self.turning_angles = None
        self.time_diffs = None
        self.bearings_abs = None
        self.mean_velocity = None
        self.diffusivity_value = None
        
    def interpolate_to_regular_timesteps(self, data, time_step):
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
            
            self.data = df_regular
            
            print(f"Interpolated data points: {len(data)}")
            print(f"Time step: {time_step}")
            
            # Calculate actual time differences to verify
            time_diffs = np.diff(data['time']).astype('timedelta64[s]').astype(float)
            print(f"Mean time interval: {np.mean(time_diffs)/3600:.2f} hours")
            print(f"Std time interval: {np.std(time_diffs)/3600:.2f} hours")
            
    

    def coordinates_to_xy(self):
        if self.data is None:
            raise ValueError("Run interpolate_to_regular_timesteps first.")

        x, y = self.proj(
            self.data["longitude"].values,
            self.data["latitude"].values
        )
        self.data["x"] = x
        self.data["y"] = y

            
            
    def compute_step_lengths(self):
        dx = np.diff(self.data["x"])
        dy = np.diff(self.data["y"])

        self.dx = dx
        self.dy = dy
        self.step_lengths = np.sqrt(dx**2 + dy**2)
        return self.dx, self.dy, self.step_lengths

    def turning_angles(self) :
        """
        Calculates turning angles = change in direction between steps (radians, -π to π), where 
        positive = left turn, negative = right turn;
        and bearings_abs = mean preferred direction in radians .
        """
        self.bearings_abs = np.arctan2(self.dy, self.dx)
        self.turning_angl = np.diff(self.bearings_abs)
        self.turning_ang= np.arctan2(np.sin(self.turning_angl), np.cos(self.turning_angl))   
        self.directional_bias = circmean(self.bearings_abs)
        return self.bearings_abs, self.turning_angl
        
    def compute_time_intervals(self):
        dates = pd.to_datetime(self.data["time"])
        self.time_diffs = np.diff(dates).astype("timedelta64[s]").astype(float)
        return self.time_diffs

    def behavioural_speed(self, use_lit_prior, prior_weight=0.5,
                        literature_cruising_min_ms = 0.67, literature_cruising_max_ms = 1.34):
        """
        Calculate mean behavioral velocity with optional literature prior. Calculate
        velocities bas                                                                                                                          ed on some predefined step_length and time-diffs. To avoid unrealistic estimate, can be optionally validated
        or meaned by literature_cruising_min_ms and literature_cruising_max_ms
        """
        velocities = self.step_lengths / self.time_diffs
        mean_velocity = np.median(velocities[velocities > 0])
        
        mean_velocity_literature = (literature_cruising_min_ms + literature_cruising_max_ms) / 2  # 1.005 m/s ≈ 3.6 km/h
            
        if use_lit_prior:
                # Weighted average between observed and literature
            mean_velocity = (1 - prior_weight) * mean_velocity + \
                            prior_weight * mean_velocity_literature
        self.mean_velocity = mean_velocity
        return self.mean_velocity
        
    def directional_persistance(self):
            """Estimate von Mises concentration parameter using MLE."""
            r = np.abs(np.mean(np.exp(1j * self.turning_angl)))
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
                print(f"Mean velocity is HIGHER than typical cruising (check for outliers)")
                status_mean = "TOO HIGH"
        
            # Check for unrealistic outliers
            outliers_high = np.sum(velocities > 5.0)  # > 18 km/h is very fast
            outliers_low = np.sum(velocities < 0.1)   # < 0.36 km/h is very slow
            
            if outliers_high > 0:
                print(f"Found {outliers_high} measurements > 5 m/s (18 km/h) - may be unrealistic")
            if outliers_low > 0:
                print(f"Found {outliers_low} measurements < 0.1 m/s (0.36 km/h) - near stationary")
            
            return {
                'mean_velocity_ms': mean_velocity,
                'mean_velocity_kmh': mean_kmh,
                'status': status_mean,
            }

    def diffusivity(self):
        """
        Calculate movement diffusivity (spread parameter).
        
        diffusivity  = Diffusion coefficient (m²/s);
        Higher = more random/spread out movement
        """
        mean_squared_displacement = np.mean(self.step_lengths**2)
        mean_time_step = np.median(self.time_diffs)
        diffusivity = mean_squared_displacement / (4 * mean_time_step)
        return diffusivity


    def compute_current_offset( x, y,  age_class, grid_x, grid_y, currents_u, currents_v, dt, alpha_pup=0.6, alpha_adult=0.4):
        """
        Compute the ocean-current displacement (meters) for a timestep dt.
        """
        # age-dependent drift parameter
        if age_class.lower() == 'pup':
            alpha = alpha_pup
        else:
            alpha = alpha_adult

        ix = np.argmin(np.abs(grid_x - x))
        iy = np.argmin(np.abs(grid_y - y))

        u = currents_u[iy, ix]    # eastward 
        v = currents_v[iy, ix]    # northward
        
        offset_x = alpha * u * dt
        offset_y = alpha * v * dt

        return offset_x, offset_y

data_path = "/home/poiosh/movement_py/shark_13_with_currents.csv"
data, bbox = shark_data_filter(data_path)
fetch_landcover_data(bbox)

from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_landmark_mapping, \
    set_forbidden_landmark
from random_walk_package.core.MixedWalker import *

studies = ["elephant_study/", "baboon_SA_study/", "leap_of_the_cat/", "Boars_Austria/", "Cranes Kazakhstan/"]

study = studies[2]
kernel_mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=5)
set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=6, diffusity=1)
set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                     step_size=5,
                     directions=1,
                     diffusity=2.6)
set_forbidden_landmark(kernel_mapping, WATER)