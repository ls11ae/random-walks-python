import matplotlib.pyplot as plt
import numpy as np

from random_walk_package.core.hmm.models import fit_data
from random_walk_package.core.hmm.utils import rotate_vector, generate_angles, to_json

def calculate_steps_cor_grouped(dt_threshold, dt_tolerance, animal_trajectories):
    steps = [[], [], []]
    count_total = 0
    count_discarded = 0

    for bettong_id, entries in animal_trajectories.items():
        cur_len = 0
        # Calculate time differences between consecutive entries
        for i in range(2, len(entries)):
            count_total += 1
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            time_diff_1 = (entries[i - 1][2] - entries[i - 2][2]).total_seconds() / 60
            if abs(time_diff_0 - dt_threshold) <= dt_threshold * dt_tolerance and abs(
                    time_diff_1 - dt_threshold) <= dt_tolerance * dt_threshold:
                # steps.append((entries[i][0]-entries[i-1][0],entries[i][1]-entries[i-1][1]))
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                c = (entries[i][0], entries[i][1])
                steps[entries[i - 1][3] - 1].append(rotate_vector(a, b, c))
                cur_len += 1
            else:
                count_discarded += 1

            # durations.append(time_diff.total_seconds()/60)  # Convert to
    # print(steps[0])
    # print(f"{min([x for x, _ in steps]), max([x for x, _ in steps]), min([y for _, y in steps]), max([y for _, y in steps])}")
    print(f"  total of {count_total} steps, {(count_discarded / count_total) * 100.0}% discarded")
    print(f" State 1 {len(steps[0])}")
    print(f" State 2 {len(steps[1])}")
    print(f" State 3 {len(steps[2])}")
    return steps

def calculate_steps_brownian_grouped(dt_threshold, dt_tolerance, animal_trajectories):
    steps = [[], [], []]
    count_total = 0
    count_discarded = 0

    for bettong_id, entries in animal_trajectories.items():
        for i in range(1, len(entries)):
            count_total += 1
            time_diff = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60

            if abs(time_diff - dt_threshold) <= dt_threshold * dt_tolerance:
                dx = entries[i][0] - entries[i - 1][0]
                dy = entries[i][1] - entries[i - 1][1]
                steps[entries[i - 1][3] - 1].append((dx, dy))
            else:
                count_discarded += 1

    print(f"  total of {count_total} steps, {(count_discarded / count_total) * 100.0}% discarded")
    print(f" State 1 {len(steps[0])}")
    print(f" State 2 {len(steps[1])}")
    print(f" State 3 {len(steps[2])}")

    return steps


updating = False

def create_and_plot_kernels(a, b, c, rnge, reso, output_dir = None):
    from random_walk_package.core.hmm.visualization import generate_heatmap

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    Za = Zb = Zc = None
    if len(a) >= 3:
        generate_heatmap(axs[0, 0], a, rnge, reso)
        Za = fit_data(axs[1, 0], a, rnge, reso)
    if len(b) >= 3:
        generate_heatmap(axs[0, 1], b, rnge, reso)
        Zb = fit_data(axs[1, 1], b, rnge, reso)
    if len(c) >= 3:
        generate_heatmap(axs[0, 2], c, rnge, reso)
        Zc = fit_data(axs[1, 2], c, rnge, reso)

    linked_pairs = [(axs[0, 0], axs[1, 0]), (axs[0, 1], axs[1, 1]), (axs[0, 2], axs[1, 2])]

    # Function to synchronize zoom/pan between pairs
    def on_xlim_changed(event_ax):
        global updating
        if updating:
            return  # Prevent recursive calls

        # Temporarily disable updating to avoid recursion
        updating = True
        # Check each pair of linked plots
        for ax1, ax2 in linked_pairs:
            if event_ax == ax1:
                # Sync ax2 with ax1
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_ylim(ax1.get_ylim())
            elif event_ax == ax2:
                # Sync ax1 with ax2
                ax1.set_xlim(ax2.get_xlim())
                ax1.set_ylim(ax2.get_ylim())

        # Redraw the figure to apply changes
        fig.canvas.draw_idle()
        updating = False

    # Connect zoom/pan event listeners for all axes in the linked pairs
    for ax1, ax2 in linked_pairs:
        ax1.callbacks.connect('xlim_changed', on_xlim_changed)
        ax1.callbacks.connect('ylim_changed', on_xlim_changed)
        ax2.callbacks.connect('xlim_changed', on_xlim_changed)
        ax2.callbacks.connect('ylim_changed', on_xlim_changed)
    if output_dir is not None:
        plt.savefig(output_dir, bbox_inches='tight')
    return Za, Zb, Zc

def pure_cor_grouped(dt_threshold, dt_tolerance, animal_trajectories, rnge, reso, out=None):
    # point clouds of relative steps per state
    a, b, c = calculate_steps_cor_grouped(dt_threshold, dt_tolerance, animal_trajectories)
    return create_and_plot_kernels(a, b, c, rnge, reso, out)

def pure_brw_grouped(dt_threshold, dt_tolerance, animal_trajectories, rnge, reso, out=None):
    a, b, c = calculate_steps_brownian_grouped(dt_threshold, dt_tolerance, animal_trajectories)
    return create_and_plot_kernels(a, b, c, rnge, reso, out)
