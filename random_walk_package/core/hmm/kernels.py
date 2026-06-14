"""Compatibility wrapper for state-kernel generation helpers."""

from kernelcma import (
    create_and_plot_kernels,
    fit_gaussian_density,
    generate_heatmap,
    pure_brw_grouped,
    pure_cor_grouped,
    rotate_vector,
)
from kernelcma.steps import calculate_steps_brownian_grouped, calculate_steps_cor_grouped

fit_data = fit_gaussian_density

__all__ = [
    "calculate_steps_brownian_grouped",
    "calculate_steps_cor_grouped",
    "create_and_plot_kernels",
    "fit_data",
    "fit_gaussian_density",
    "generate_heatmap",
    "pure_brw_grouped",
    "pure_cor_grouped",
    "rotate_vector",
]
