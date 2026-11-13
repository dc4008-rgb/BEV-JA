"""Detached five-step FOA statistics; no gradients, models, or random draws.

Deficits are supplied by the identity-view objective helpers. In particular,
this controller does not clip depth deficits, or implement detector d_max.
Call observe after the attack update. Its new weights apply to the following
cycle, never retrospectively to the step whose deficits were observed.
"""
from dataclasses import dataclass
import math
from numbers import Real


INNER_STEPS = 25
GROUPS = ("detector", "depth", "task")


@dataclass(frozen=True)
class PeriodicConfig:
    period: int = 5
    model_mu: float = 0.5
    task_mu: float = 0.8
    model_temperature: float = 1.0
    task_temperature: float = 1.0
    ema_decay: float = 0.7
    ratio_delta: float = 0.001
    ratio_min: float = 0.5
    ratio_max: float = 2.0
    model_adaptive: bool = True
    task_adaptive: bool = True
    ema: bool = True

    def __post_init__(self):
        if type(self.period) is not int or self.period not in (1, 5):
            raise ValueError("period must be exactly 1 or 5")
        for name in ("model_mu", "task_mu", "ema_decay"):
            value = _finite_number(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(name + " must be in [0, 1]")
        for name in ("model_temperature", "task_temperature", "ratio_delta",
                     "ratio_min", "ratio_max"):
            if _finite_number(getattr(self, name), name) <= 0.0:
                raise ValueError(name + " must be positive")
        if self.ratio_min > self.ratio_max:
            raise ValueError("ratio_min must not exceed ratio_max")
        for name in ("model_adaptive", "task_adaptive", "ema"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be a boolean")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _pair(value, name):
    # Tensor conversion is detached before either device transfer or extraction.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " requires exactly two scalar deficits")
    result = []
    for item in value:
        if hasattr(item, "detach"):
            item = item.detach()
        if hasattr(item, "cpu"):
            item = item.cpu()
        if hasattr(item, "item"):
            item = item.item()
        number = _finite_number(item, name)
        if number < 0.0:
            raise ValueError(name + " must be nonnegative")
        result.append(number)
    return result


def _initial_weights():
    return {"detector": [1.0, 1.0], "depth": [0.5, 0.5],
            "task": [0.5, 0.5]}


def _copy_groups(value):
    return {name: list(value[name]) for name in GROUPS}


def _mean(values):
    # Divide before summing to avoid overflow for otherwise finite means.
    count = len(values)
    result = math.fsum(value / count for value in values)
    return _finite_number(result, "mean deficit")


def _softmax_pair(scores, temperature):
    # Subtract before division: even large finite scores stay well behaved.
    largest = max(scores)
    values = [math.exp((value - largest) / temperature) for value in scores]
    first = values[0] / math.fsum(values)
    return [first, 1.0 - first]


class PeriodicController:
    """State scoped to an explicitly reset 25-step outer loop.

    Cumulative counts cover this controller instance; instantiate a new
    controller per sample to obtain per-sample 100-cycle/80-refresh counts.
    Returned weights and cycle logs are independent plain-Python snapshots.
    """

    def __init__(self, config):
        if not isinstance(config, PeriodicConfig):
            raise ValueError("config must be PeriodicConfig")
        self.config = config
        self._outer_index = None
        self._last_step = 0
        self._current = _initial_weights()
        self._previous_D = None
        self._members = []
        self._cycle_count_outer = 0
        self._refresh_count_outer = 0
        self._cumulative_cycles = 0
        self._cumulative_refreshes = 0

    def reset_outer(self, outer_index):
        if type(outer_index) is not int or outer_index < 0:
            raise ValueError("outer_index must be a nonnegative integer")
        self._outer_index = outer_index
        self._last_step = 0
        self._current = _initial_weights()
        self._previous_D = None
        self._members = []
        self._cycle_count_outer = 0
        self._refresh_count_outer = 0

    def weights(self):
        return _copy_groups(self._current)

    def observe(self, inner_step, detection_deficits, depth_deficits):
        if self._outer_index is None:
            raise ValueError("reset_outer is required before observe")
        if (type(inner_step) is not int or not 1 <= inner_step <= INNER_STEPS
                or inner_step != self._last_step + 1):
            raise ValueError("inner steps must be consecutive integers 1..25")
        member = {"step": inner_step,
                  "detector": _pair(detection_deficits, "detection_deficits"),
                  "depth": _pair(depth_deficits, "depth_deficits")}
        members = self._members + [member]
        if inner_step % self.config.period:
            self._members = members
            self._last_step = inner_step
            return None

        D = {group: [_mean([row[group][i] for row in members])
                     for i in range(2)] for group in ("detector", "depth")}
        # Task statistics are unweighted model means, independent of w and p.
        D["task"] = [_mean(D[group]) for group in ("detector", "depth")]
        R = {}
        A = {}
        for group in GROUPS:
            R[group] = [1.0, 1.0] if self._previous_D is None else [
                self._ratio(D[group][i], self._previous_D[group][i])
                for i in range(2)]
            A[group] = [_finite_number(D[group][i] * R[group][i],
                                       "deficit-progress score")
                        for i in range(2)]

        current = self.weights()
        target = _initial_weights()
        if self.config.model_adaptive:
            det = _softmax_pair(A["detector"], self.config.model_temperature)
            depth = _softmax_pair(A["depth"], self.config.model_temperature)
            mu = self.config.model_mu
            target["detector"] = [(1.0 - mu) + 2.0 * mu * x for x in det]
            target["depth"] = [(1.0 - mu) / 2.0 + mu * x for x in depth]
        if self.config.task_adaptive:
            task = _softmax_pair(A["task"], self.config.task_temperature)
            mu = self.config.task_mu
            target["task"] = [(1.0 - mu) / 2.0 + mu * x for x in task]
        next_weights = _copy_groups(target)
        for group in GROUPS:
            enabled = (self.config.task_adaptive if group == "task"
                       else self.config.model_adaptive)
            if not enabled:
                # An off branch is exactly its initial value, without EMA drift.
                next_weights[group] = _initial_weights()[group]
            elif self.config.ema:
                rho = self.config.ema_decay
                next_weights[group] = [rho * current[group][i]
                    + (1.0 - rho) * target[group][i] for i in range(2)]

        applied_next = inner_step < INNER_STEPS
        cycle_count = self._cycle_count_outer + 1
        refresh_count = self._refresh_count_outer + int(applied_next)
        cumulative_cycles = self._cumulative_cycles + 1
        cumulative_refreshes = self._cumulative_refreshes + int(applied_next)
        cycle = {"schema": "periodic-foa-cycle-v1",
                 "outer_index": self._outer_index,
                 "cycle_index": cycle_count,
                 "period": self.config.period,
                 "member_steps": [row["step"] for row in members],
                 "D": D, "R": R, "A": A, "target": target,
                 "current": current, "next": next_weights,
                 "applied_next": applied_next,
                 "cycle_count_outer": cycle_count,
                 "refresh_count_outer": refresh_count,
                 "cumulative_cycles": cumulative_cycles,
                 "cumulative_refreshes": cumulative_refreshes}
        # Commit only after all inputs and calculations have been validated.
        self._previous_D = _copy_groups(D)
        self._members = []
        self._last_step = inner_step
        self._cycle_count_outer = cycle_count
        self._refresh_count_outer = refresh_count
        self._cumulative_cycles = cumulative_cycles
        self._cumulative_refreshes = cumulative_refreshes
        if applied_next:
            self._current = _copy_groups(next_weights)
        return cycle

    def _ratio(self, current, previous):
        delta = self.config.ratio_delta
        # Dividing scaled positive terms avoids overflow in D + delta.
        scale = max(current, previous, delta)
        numerator = current / scale + delta / scale
        denominator = previous / scale + delta / scale
        if denominator == 0.0:
            return float(self.config.ratio_max)
        ratio = numerator / denominator
        return min(float(self.config.ratio_max),
                   max(float(self.config.ratio_min), ratio))
