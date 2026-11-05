"""Allocation methods for the FairShare educational game-theory demo."""
import numpy as np
from scipy.optimize import minimize


def allocate_nash(capacity, demands, weights):
    """
    Maximize the Nash product of utility gains:
        product_i [u_i(x_i) - u_i(0)]
    with u_i(x) = ln(1 + weight_i * x), disagreement allocation x_i=0.

    Numerically maximize sum(log(u_i(x_i))) subject to:
        0 <= x_i <= demand_i and sum(x_i) <= capacity.
    """
    demands = np.asarray(demands, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if demands.ndim != 1 or weights.ndim != 1 or len(demands) != len(weights):
        raise ValueError("Demands and weights must be 1-D arrays of equal length.")
    if capacity < 0 or np.any(demands < 0) or np.any(weights <= 0):
        raise ValueError("Capacity/demands must be non-negative and weights positive.")
    if len(demands) == 0:
        return np.array([], dtype=float)
    if capacity == 0 or np.sum(demands) == 0:
        return np.zeros_like(demands)
    if demands.sum() <= capacity:
        return demands.copy()

    # Small positive lower bound avoids log(0) in the Nash objective.
    active = demands > 0
    result = np.zeros_like(demands)
    if not np.any(active):
        return result
    d = demands[active]
    w = weights[active]
    eps = min(1e-7, capacity / (1000 * len(d)))
    lower = np.minimum(d, eps)
    remaining_capacity = max(0.0, capacity - lower.sum())

    def objective(x):
        # Utility gain above disagreement utility (0) is log(1 + w*x).
        utility = np.log1p(w * x)
        return -np.sum(np.log(np.maximum(utility, 1e-15)))

    x0 = np.minimum(d, capacity / len(d))
    if x0.sum() > capacity:
        x0 *= capacity / x0.sum()
    x0 = np.maximum(x0, lower)
    if x0.sum() > capacity:
        x0 = lower.copy()
        x0 *= capacity / x0.sum()

    opt = minimize(
        objective, x0, method="SLSQP",
        bounds=list(zip(lower, d)),
        constraints=[{"type": "ineq", "fun": lambda x: capacity - np.sum(x)}],
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    if not opt.success and not np.isfinite(opt.fun):
        raise RuntimeError("Nash optimization did not converge. Try different inputs.")
    x = np.clip(opt.x, 0, d)
    # Numerical cleanup: scale down if optimizer exceeds capacity by tiny rounding.
    if x.sum() > capacity:
        x *= capacity / x.sum()
    result[active] = x
    return result


def allocate_equal(capacity, demands):
    """Progressively share capacity equally, stopping when a demand is met."""
    demands = np.asarray(demands, dtype=float)
    result = np.zeros_like(demands)
    remaining = max(0.0, float(capacity))
    active = set(np.where(demands > 0)[0].tolist())
    while active and remaining > 1e-10:
        share = remaining / len(active)
        satisfied = [i for i in active if demands[i] - result[i] <= share + 1e-10]
        if not satisfied:
            for i in active:
                result[i] += share
            remaining = 0.0
        else:
            for i in satisfied:
                amount = demands[i] - result[i]
                result[i] += amount
                remaining -= amount
                active.remove(i)
    return result


def allocate_priority(capacity, demands, weights):
    """Greedy baseline: serve highest weight first until its demand is met."""
    demands = np.asarray(demands, dtype=float)
    weights = np.asarray(weights, dtype=float)
    result = np.zeros_like(demands)
    remaining = max(0.0, float(capacity))
    for i in np.argsort(-weights, kind="stable"):
        amount = min(demands[i], remaining)
        result[i] = amount
        remaining -= amount
        if remaining <= 1e-10:
            break
    return result


def fairness_index(allocation):
    """Jain's fairness index in [0, 1], computed over nonnegative allocations."""
    x = np.asarray(allocation, dtype=float)
    if len(x) == 0 or np.sum(x * x) == 0:
        return 1.0
    return float(np.sum(x) ** 2 / (len(x) * np.sum(x * x)))


def build_demo_dataframe(names, demands, allocation):
    return [{"Process": n, "Demand": float(d), "Allocated": float(a)}
            for n, d, a in zip(names, demands, allocation)]
