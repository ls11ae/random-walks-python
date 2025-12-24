import struct

import pandas as pd


def serialize_env_grid(binary_dir, kernel_df: pd.DataFrame, time_col, env_samples: int, T: int):
    DT_FMT = "<4i"  # 4 ints
    KP_FMT = "<?qqfqq"  # bool, 2 x size_t, float, 2 x size_t
    DIMS_FMT = "<qqq"  # 3 x size_t
    with open(binary_dir, "wb") as f:
        # write dimensions
        f.write(struct.pack(
            DIMS_FMT,
            env_samples,
            env_samples,
            T
        ))

        for y in range(env_samples):
            for x in range(env_samples):
                cell = kernel_df[(kernel_df["y"] == y) & (kernel_df["x"] == x)]

                for _, row in cell.iterrows():
                    t = row[time_col]

                    # DateTime
                    f.write(struct.pack(
                        DT_FMT,
                        t.year,
                        t.month,
                        t.day,
                        t.hour
                    ))

                    # KernelParameters
                    f.write(struct.pack(
                        KP_FMT,
                        bool(row["is_brownian"]),
                        int(row["S"]),
                        int(row["D"]),
                        float(row["diffusity"]),
                        int(row["bias_x"]),
                        int(row["bias_y"])
                    ))

                    # landmark
                    f.write(struct.pack("<i", int(row["terrain"])))
