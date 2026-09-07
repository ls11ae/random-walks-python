from enum import Enum, IntEnum

import pandas as pd

from randomwalks.bindings.data_structures.Terrain import Animal, BarrierMode, MesaLandcover
from randomwalks.bindings.data_structures.types import Reachability
from randomwalks.core.WalkerHelper import WalkerHelper


class UnmodelledStatePolicy(str, Enum):
    SKIP = "skip"
    PREVIOUS = "previous"


def _coerce_animal(animal_type):
    if isinstance(animal_type, Animal):
        return animal_type
    if isinstance(animal_type, IntEnum):
        return Animal(int(animal_type))
    if isinstance(animal_type, str):
        return Animal[animal_type.upper()]
    return Animal(animal_type)


def _validate_interpolation_stride(n):
    return WalkerHelper.positive_integer(n, "n")


def _every_nth_point(steps, n):
    """Select interpolation endpoints strictly by zero-based row position."""
    n = _validate_interpolation_stride(n)
    if n == 1 or len(steps) <= 1:
        return steps.copy()
    return steps.iloc[::n].copy()


def _interpolation_stride_folder(n):
    n = _validate_interpolation_stride(n)
    return None if n == 1 else f"every_{n}{_ordinal_suffix(n)}_point"


def _ordinal_suffix(value):
    value = int(value)
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _default_barriers(animal):
    """Return preset terrain barriers for an animal type."""
    return [MesaLandcover.TREE_COVER] if animal == Animal.MARINE else []


def _default_barrier_mode(animal):
    """Marine land is forbidden; airborne movement uses full reachability."""
    if animal == Animal.AIRBORNE:
        return BarrierMode.ALLOW
    return BarrierMode.FORBID if animal == Animal.MARINE else BarrierMode.AVOID


def _resolve_barrier_mode(animal, barrier_mode):
    if barrier_mode is None:
        return _default_barrier_mode(animal)
    return BarrierMode(barrier_mode)


def _reachability_for_barrier_mode(barrier_mode):
    return {
        BarrierMode.FORBID: Reachability.HARD,
        BarrierMode.AVOID: Reachability.RELAXED,
        BarrierMode.ALLOW: Reachability.FULL,
    }[BarrierMode(barrier_mode)]


def _reachability_for_segment(barrier_mode, barriers, terrain_values):
    """Avoid reachability masks when no configured barrier is present.

    State-dependent kernels do not currently apply terrain transition weights.
    Therefore HARD/SOFT and FULL reachability are equivalent unless the
    segment contains a terrain value configured as a barrier. This distinction
    matters for large kernels because pooled masked contexts cache full kernel
    copies for distinct boundary/barrier masks.
    """
    mode = BarrierMode(barrier_mode)
    if mode == BarrierMode.ALLOW:
        return Reachability.FULL

    configured_barriers = {int(value) for value in barriers or []}
    present_values = {int(value) for value in terrain_values}
    if configured_barriers.isdisjoint(present_values):
        return Reachability.FULL
    return _reachability_for_barrier_mode(mode)


def _hard_barrier_endpoints(terrain, barriers, start_point, end_point, barrier_mode):
    """Return endpoint labels that fall on barriers in strict reachability.

    Observations can be assigned to a neighbouring land node when a coastline
    is rasterised.  A HARD walk cannot consistently start from or condition on
    such a node, so callers can skip that endpoint pair instead of asking the
    native solver to manufacture an impossible path.  RELAXED and FULL modes
    deliberately do not use this guard.
    """
    if BarrierMode(barrier_mode) != BarrierMode.FORBID:
        return ()
    forbidden = {int(value) for value in barriers or []}
    if not forbidden:
        return ()
    labels = []
    for label, (x, y) in (("start", start_point), ("end", end_point)):
        if int(terrain.at(int(x), int(y))) in forbidden:
            labels.append(label)
    return tuple(labels)


def _validate_state_fill_gap(policy, max_state_fill_gap):
    if policy is UnmodelledStatePolicy.SKIP:
        return None
    if max_state_fill_gap is None:
        raise ValueError("max_state_fill_gap is required when unmodelled_state_policy='previous'.")
    gap = pd.Timedelta(max_state_fill_gap)
    if gap <= pd.Timedelta(0):
        raise ValueError("max_state_fill_gap must be positive.")
    return gap


def _validate_max_time_steps(max_time_steps):
    if max_time_steps is None:
        return None
    try:
        return WalkerHelper.positive_integer(max_time_steps, "max_time_steps")
    except ValueError as exc:
        raise ValueError(
            "max_time_steps must be a positive integer or None, "
            f"got {max_time_steps!r}."
        ) from exc


def _state_walk_progress_message(
        animal_id,
        animal_index,
        animal_count,
        pair_index,
        pair_count,
        *,
        state=None,
        S=None,
        T=None,
        status=None,
):
    if hasattr(state, "item"):
        state = state.item()
    state_text = "NA" if state is None else str(state)
    s_text = "NA" if S is None else str(int(S))
    t_text = "NA" if T is None else str(int(T))
    message = (
        f"{animal_id} [animal {animal_index}/{animal_count}] "
        f"endpoint pair {pair_index}/{pair_count} | state={state_text}, S={s_text}, T={t_text}"
    )
    return f"{message} ({status})" if status else message


__all__ = ["UnmodelledStatePolicy"]
