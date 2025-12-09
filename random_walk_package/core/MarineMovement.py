import numpy as np
import pandas as pd
from scipy.stats import circmean

from random_walk_package import AnimalMovementProcessor


def shark_data_filter(data_path):
    data = pd.read_csv(data_path)
    data = data.copy()
    data = data.sort_values("time")
    data["time"] = pd.to_datetime(data["time"]).dt.normalize()
    data["time"] = data["time"] + pd.Timedelta(hours=12)
    data = data.drop_duplicates(subset=["time"], keep="first")
    data["tag-local-identifier"] = data["ptt"]
    data["location-long"] = data["longitude"]
    data["location-lat"] = data["latitude"]
    data['timestamp'] = data["time"]

    return data


class MarineMovement:
    def __init__(self, age_class="adult", data=None):
        self.age_class = age_class.lower()

        # Projection: azimuthal equidistant
        # self.proj = Proj(proj="utm") #lat_0=0, lon_0=0

        # Results stored
        self.data = data
        self.processor = AnimalMovementProcessor(df=data)
        self.dx = None
        self.dy = None
        self.step_lengths = None
        self.turning_angl = None
        self.time_diffs = None
        self.bearings_abs = None
        self.mean_velocity = None
        self.diffusivity_value = None
        self.directional_bias = None

    def terrain_data(self):
        return self.processor.create_landcover_data_txt(resolution=250)

    def coordinates_to_xy(self):
        # self.data = data
        # for aid, bbox in self.processor.bbox:

        '''x, y = self.proj(
            self.data["longitude"].values,
            self.data["latitude"].values
        )
        '''
        self.terrain_data()
        _, coordinates, _ = self.processor.create_movement_data(
            samples=-1)  # if i have a lot of enties with smal dt, takes every 100th entry
        steps = next(iter(coordinates.values()))
        self.data["x"] = [t[0] for t in steps]
        self.data["y"] = [t[1] for t in steps]

    def compute_step_lengths(self):
        dx = np.diff(self.data["x"])
        dy = np.diff(self.data["y"])

        self.dx = dx
        self.dy = dy
        self.step_lengths = np.sqrt(dx ** 2 + dy ** 2)
        return self.dx, self.dy, self.step_lengths

    def turning_angles(self):
        """
        Calculates turning angles = change in direction between steps (radians, -π to π), where 
        positive = left turn, negative = right turn;
        and bearings_abs = mean preferred direction in radians .
        """
        self.bearings_abs = np.arctan2(self.dy, self.dx)
        raw = np.diff(self.bearings_abs)
        self.turning_angl = np.arctan2(np.sin(raw), np.cos(raw))
        if len(self.bearings_abs) > 0:
            self.directional_bias = circmean(self.bearings_abs)
        else:
            self.directional_bias = np.nan
        return self.bearings_abs, self.turning_angl

    def compute_time_intervals(self):
        dates = pd.to_datetime(self.data["time"])
        self.time_diffs = np.diff(dates).astype("timedelta64[s]").astype(float)
        return self.time_diffs

    def behavioural_speed(self, use_lit_prior=True, prior_weight=0.5,
                          literature_cruising_min_ms=0.67, literature_cruising_max_ms=1.34):
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
        r = np.abs(np.mean(np.exp(1j * self.turning_angl)))  # now its taking each angle,
        if r < 0.53:
            kappa = 2 * r + r ** 3 + 5 * r ** 5 / 6
        elif r < 0.85:
            kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            kappa = 1 / (r ** 3 - 4 * r ** 2 + 3 * r)
        return r, kappa

    def compare_with_reference_speeds(self, velocities, mean_velocity):
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
        outliers_low = np.sum(velocities < 0.1)  # < 0.36 km/h is very slow

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
        mean_squared_displacement = np.mean(self.step_lengths ** 2)
        mean_time_step = np.median(self.time_diffs)
        diffusivity = mean_squared_displacement / (4 * mean_time_step)
        return diffusivity

    def fetch_ocean_data(self, output_directory: str, dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"):
        "This function is fetching the data about northward and eastward ocean currents with the hardcoded dataset_id from copernicus "
        import copernicusmarine
        max_lat = self.data["location-lat"].max()
        max_long = self.data["location-long"].max()
        min_lat = self.data["location-lat"].min()
        min_long = self.data["location-long"].min()
        start_date = self.data["timestamp"].min()
        end_data = self.data["timestamp"].max()
        get_result_annualmean = copernicusmarine.subset(
            dataset_id=dataset_id,
            output_directory=output_directory,
            file_format="csv",
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_longitude=min_long,
            maximum_longitude=max_long,
            start_datetime=start_date,
            end_datetime=end_data,
            variables=["time", "latitude", "longitude", "uo", "vo"]
        )

        print(f"List of saved files: {get_result_annualmean}")

    def compute_current_offset(x, y, age_class, grid_x, grid_y, currents_u, currents_v, dt, alpha_pup=0.6,
                               alpha_adult=0.4):
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

        u = currents_u[iy, ix]  # eastward
        v = currents_v[iy, ix]  # northward

        offset_x = alpha * u * dt
        offset_y = alpha * v * dt

        return offset_x, offset_y


data_path = "/home/poiosh/movement_py/shark_13_with_currents.csv"
data = shark_data_filter(data_path)

model = MarineMovement(data=data, age_class="pup")

model.coordinates_to_xy()

dx, dy, steps = model.compute_step_lengths()
bear, turning = model.turning_angles()
dt = model.compute_time_intervals()

mean_v = model.behavioural_speed()
D = model.diffusivity()
r, kappa = model.directional_persistance()

print(dx, "dy", dy, "steps", steps, "bear", bear, "turning", turning, "dt", dt, "mean_v", mean_v, "diffusity", D, "r",
      r, "kappa", kappa)
print(data.head())

# although filtering of dates also happens in C, it makes sense to set the dates of the interval of the study here

processor = AnimalMovementProcessor(time_col='timestamp', lon_col='longitude', lat_col='latitude', data=data,
                                    id_col="tag-local-identifier", crs="EPSG:4326")
animal_to_traj = processor.create_movement_data_dict()
print(animal_to_traj)
