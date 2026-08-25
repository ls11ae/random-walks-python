import numpy as np


class WalkerHelper:
    @staticmethod
    def validate_point(point, width, height, *, name="point"):
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{name} ({x}, {y}) out of bounds for grid {width}x{height}")
        return x, y

    @staticmethod
    def validate_steps(steps, width, height):
        if steps is None or len(steps) < 2:
            raise ValueError("At least two steps are required")
        return [WalkerHelper.validate_point(step, width, height, name="step") for step in steps]

    @staticmethod
    def validate_dimensions(width, height, timesteps, step_size=None, directions=None):
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid grid dimensions: {width}x{height}")
        if timesteps <= 0:
            raise ValueError(f"Invalid time steps: {timesteps}")
        if step_size is not None and step_size <= 0:
            raise ValueError(f"Invalid step size: {step_size}")
        if directions is not None and directions <= 0:
            raise ValueError(f"Invalid directions: {directions}")

    @staticmethod
    def append_segment(full_path, segment):
        segment = np.asarray(segment, dtype=np.int64)
        if segment.size == 0:
            return full_path
        if full_path.size == 0:
            return segment
        return np.vstack((full_path, segment[1:]))

    @staticmethod
    def create_timed_df(
            steps_df,
            geodetic_path_df,
            animal_id,
            idx,
            segment_boundaries,
            *,
            traj_id_col="traj_id",
            time_col="time",
    ):
        rows = []
        if len(idx) < 2:
            return rows

        import pandas as pd

        for i in range(len(idx) - 1):
            start = segment_boundaries[i] if i < len(segment_boundaries) else 0
            stop = segment_boundaries[i + 1] if i + 1 < len(segment_boundaries) else len(geodetic_path_df)
            if i == len(idx) - 2:
                stop = len(geodetic_path_df)

            segment_df = geodetic_path_df.iloc[start:stop].copy()
            if segment_df.empty:
                continue

            times = pd.date_range(
                start=idx[i],
                end=idx[i + 1],
                periods=len(segment_df),
            )
            segment_df[traj_id_col] = animal_id
            segment_df[time_col] = times
            rows.append(segment_df)

        return rows

    @staticmethod
    def direction_from_points(start_x, start_y, end_x, end_y, directions):
        dx = end_x - start_x
        dy = end_y - start_y
        if dx == 0 and dy == 0:
            return 0
        angle = np.arctan2(dy, dx)
        normalized = (angle + 2 * np.pi) % (2 * np.pi)
        return int(np.round(normalized / (2 * np.pi) * directions)) % directions

    @staticmethod
    def resample_kernel_to_grid(kernel, step_size):
        from skimage.transform import resize

        target = 2 * int(step_size) + 1
        resampled = resize(
            kernel,
            (target, target),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        resampled = np.maximum(resampled, 0)
        total = resampled.sum()
        if total > 0:
            resampled /= total
        return resampled

    @staticmethod
    def coerce_grid_walk(walk):
        if walk is None:
            return None
        try:
            array = np.asarray(walk, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            return None
        if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
            return None
        return [(int(x), int(y)) for x, y in array]

    @staticmethod
    def normalize_grid_walk(walk, start, end):
        coerced = WalkerHelper.coerce_grid_walk(walk)
        if coerced is not None:
            return coerced
        return [
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
        ]

    @staticmethod
    def validate_grid_paths(paths, width, height, label="path"):
        for path in paths or []:
            for x, y in path:
                WalkerHelper.validate_point((x, y), width, height, name=f"{label} coordinate")

    @staticmethod
    def runtime_kernel(
            base_kernel,
            step_size,
            cell_size,
            source_range,
            *,
            externally_supplied,
            kernel_range_m=None,
    ):
        """Map a physical kernel onto the policy-resolved RW grid radius."""
        if externally_supplied and kernel_range_m is not None:
            if not np.isfinite(kernel_range_m) or float(kernel_range_m) <= 0:
                raise ValueError("A positive finite kernel_range_m is required for a physical kernel.")
            if not np.isfinite(cell_size) or float(cell_size) <= 0:
                raise ValueError("A positive finite RW cell size is required to resample a physical kernel.")
            step_size = max(1, int(np.ceil(float(kernel_range_m) / float(cell_size))))
        del source_range
        return WalkerHelper.resample_kernel_to_grid(base_kernel, step_size)

    @staticmethod
    def validate_policy_resolution(T, S):
        return tuple(
            WalkerHelper.positive_integer(value, name)
            for name, value in (("T", T), ("S", S))
        )

    @staticmethod
    def positive_integer(value, name="value"):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        return int(numeric)

    @staticmethod
    def optional_positive_integer(value, name="value"):
        if value is None:
            return None
        return WalkerHelper.positive_integer(value, name)
