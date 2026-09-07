~~~bash
python -m venv myenv
~~~

~~~bash
source myenv/bin/activate
~~~

~~~bash
pip install .
~~~

~~~bash
cp build/lib/librandom_walk.so random_walk_package/
~~~

~~~bash
python -m tests.test
~~~

## `adehabitatHR` reference comparison

The animal-wise R/BRB comparison runner, including 50% and 95% isopleth plots
and guidance for matching an arbitrary Python random-walk run where the models
permit it, is documented in
[`R_adehabitatHR_reference/README.md`](R_adehabitatHR_reference/README.md).

## Native-time state-kernel movement

Use `AdaptiveKernelMovementPolicy` when fitted state kernels represent a
physical model interval. The policy maps each state's complete physical kernel
range onto the realized local random-walk grid and derives an uncapped
transition count from each observed interval:

```python
movement_policy = AdaptiveKernelMovementPolicy()

with StateDependentWalker(
    data=trajectories,
    animal_type=Animal.TERRESTRIAL,
    resolution=100,
    out_directory=output_directory,
    movement_policy=movement_policy,
    barriers=barriers,
    n=10,
) as walker:
    walker.annotate_behavior(...)
    walker.get_kernels(..., is_brownian=False)
    walker.generate_utilization_distribution(max_cell_size=5)
```

`n` controls only the observed endpoints used by random-walk interpolation and
UD generation. Its default is `1` (all points). For example, `n=10` uses points
0, 10, 20, and so on. Behavioral annotation and kernel fitting continue to use
the complete trajectory. Runs with `n > 1` are isolated under setting-specific
output directories such as
`ud_plots/every_10th_point/` and `walks/every_10th_point/`; the default run
continues to use `ud_plots/` and `walks/` directly.

Geographic UD outputs retain the complete original fixes as a separate white
point layer, even when `n > 1`; sampled random-walk paths remain separate. For
marine walkers, both the static UD PNG and interactive Leaflet map also draw a
coastline above the density surface. The coastline comes from the same land
geometry used to classify `TreeCover` barrier cells, so the visual boundary
and random-walk barrier have one source.

For regularly sampled data, using every `n`th endpoint gives an initial
`T = n`. For example, one-minute training fixes with `n=5` retain the native
one-minute correlated kernel and use five transitions between sampled
endpoints.
For a physical kernel radius `R` and realized cell size `c`, `S = ceil(R/c)`.
If `T` physical steps cannot reach the observed endpoint, the policy increases
`T` to `ceil(grid_distance/S)`. Neither value is capped, and the interpolation
loop does not subsequently alter them or the kernel's physical support.

Correlated kernels are always fitted from the original displacements at the
native sampling interval inferred from the full trajectory. They are never
linearly divided or multiplied to manufacture a different interval. The
native interval is stored as kernel metadata and used by the adaptive policy;
passing a different `dt_model_s` for a correlated fit raises `ValueError`.
Brownian fitting may still request an explicit `dt_model_s`.
