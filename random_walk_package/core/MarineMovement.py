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
    data = data.rename(columns={
        "ptt": "tag-local-identifier",
        "longitude": "location-long",
        "latitude": "location-lat",
        "time": "timestamp"
    })

    return data


class MarineMovement:
    def __init__(self, age_class="adult", data=None, resolution=1000):
        self.age_class = age_class.lower()

        # Projection: azimuthal equidistant
        # self.proj = Proj(proj="utm") #lat_0=0, lon_0=0

        # Results stored
        self.data = data
        self.processor = AnimalMovementProcessor(data=self.data, time_col="timestamp", lon_col="location-long",
                                                 lat_col="location-lat", id_col="tag-local-identifier", crs="EPSG:4326")
        self.dx = None
        self.dy = None
        self.step_lengths = None
        self.turning_angl = None
        self.time_diffs = None
        self.bearings_abs = None
        self.mean_velocity = None
        self.diffusivity_value = None
        self.directional_bias = None
        self.resolution = resolution

    def terrain_data(self):
        return self.processor.create_landcover_data_txt(is_marine=True, resolution=1000)

    def coordinates_to_xy(self):
        # self.data = data
        # for aid, bbox in self.processor.bbox:

        '''x, y = self.proj(
            self.data["longitude"].values,
            self.data["latitude"].values
        )
        '''
        self.terrain_data()
        data_dict = self.processor.create_movement_data_dict()
        steps = next(iter(data_dict.values()))
        self.data["x"] = steps.df["geo_x"].to_list()
        self.data["y"] = steps.df["geo_y"].to_list()

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
        
        Returns:
        - bearings_abs: absolute bearings at each step (radians)
        - turning_angl: turning angles between consecutive steps (radians)
        Also sets:
        - directional_bias: circular mean bearing (preferred direction)
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
        dates = pd.to_datetime(self.data["timestamp"])
        self.time_diffs = np.diff(dates).astype("timedelta64[s]").astype(float)
        return self.time_diffs

    def effective_speed(self, use_lit_prior=True, prior_weight=0.5,
                        literature_cruising_min_ms=0.67, literature_cruising_max_ms=1.34):
        """
        Calculate mean behavioral velocity with optional literature prior. Calculate
        velocities based on some predefined step_length and time-diffs. To avoid unrealistic estimate, can be optionally validated
        or meaned by literature_cruising_min_ms and literature_cruising_max_ms
        """
        velocities = self.step_lengths / self.time_diffs
        mean_velocity = np.median(velocities[velocities > 0])

        mean_velocity_literature = (literature_cruising_min_ms + literature_cruising_max_ms) / 2  # 1.005 m/s ≈ 3.6 km/h

        if use_lit_prior:
            # Weighted average between observed and literature
            mean_velocity = (1 - prior_weight) * mean_velocity + \
                            prior_weight * mean_velocity_literature
            velocities = (1 - prior_weight) * velocities + prior_weight * mean_velocity_literature
        self.mean_velocity = mean_velocity
        self.velocities = velocities
        return self.mean_velocity, self.velocities

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

    def compute_current_offset(self, x, y, grid_x, grid_y, currents_u, currents_v, dt):
        """
        ASSUMES TRAVELING STATE i.e current-positive, not avoiding or actively resisting 
        Compute the ocean-current displacement (meters) for a timestep dt. A systematic displacement caused by
        the ocean currents that affects the shark’s trajectory, added on top of the shark’s swimming movement.
        
        Parameters
        ----------
        x, y : float
            Current location
        bearing : float
            Heading in radians
        grid_x, grid_y : arrays
            Grid coordinates for currents
        currents_u, currents_v : 2D arrays
            Eastward and northward current speeds (m/s)
        dt : float
            Time step (s)
        swim_speed : float
            Shark swimming speed along its heading (m/s). Default 0.0
        
        Returns
        -------
        dx, dy : float
            Displacement in meters for this time step
            
        """

        ix = np.argmin(np.abs(grid_x - x))
        iy = np.argmin(np.abs(grid_y - y))

        u = currents_u[iy, ix]  # eastward
        v = currents_v[iy, ix]  # northward
        _, effective_speed = self.effective_speed()
        bearing, _ = self.turning_angles
        swim_x = effective_speed * np.cos(
            bearing)  # The shark moves in the direction it intends to go, rather than perpendicular or randomly, at some swimming speed.
        swim_y = effective_speed * np.sin(bearing)

        offset_x = (u + swim_x) * dt
        offset_y = (v + swim_y) * dt

        return offset_x, offset_y

    def oceanic_kernel_resolver(self, landmark, row, mode_directions="dynamic", is_brownian=False, min_D=1, max_D=36):

        bias_x, bias_y = self.compute_current_offset()  # Movement charcteristics
        effective_speed = self.effective_speed()
        bear, turning = self.turning_angles()
        dt = self.compute_time_intervals()
        diffusivity = self.diffusivity()
        r, kappa = self.directional_persistance()
        _, _, step_lengths = self.compute_step_lengths()
        S = np.mean(step_lengths)
        S = int(np.ceil(S / self.resolution))
        # set a flag if you want to calculate the number of directions at each step or globally for the whole trajectory
        abs_turning = np.abs(turning)
        abs_turning[abs_turning < 1e-6] = 1e-6
        mean_turn = np.median(abs_turning)

        def D_calc_step(turning_divisor):
            D_raw = 2 * np.pi / turning_divisor
            D_estimated = 2 ** np.ceil(np.log2(D_raw))  # # Round UP to next power of 2
            D_estimated = np.clip(D_estimated, min_D, max_D)
            return D_estimated.astype(int)

        if mode_directions == "dynamic":
            D = D_calc_step(abs_turning)

        if mode_directions == "static":
            D = D_calc_step(mean_turn)

        if is_brownian:
            D = 1

        return [is_brownian, S, D, diffusivity, bias_x, bias_y]


data_path = "/home/poiosh/movement_py/shark_13_with_currents.csv"
data = shark_data_filter(data_path)
print(data.head())
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

# fetch_ocean_data(data=data,output_directory="ocean_data.csv")
