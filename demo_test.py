"""Quick correctness checks. Run with: python demo_test.py"""
import numpy as np
from fairshare import allocate_nash, allocate_equal, allocate_priority, fairness_index

demands = np.array([60.0, 50.0, 40.0])
weights = np.array([3.0, 2.0, 1.0])
capacity = 100.0

for method in (allocate_nash,):
    allocation = method(capacity, demands, weights)
    assert np.all(allocation >= -1e-6), allocation
    assert np.all(allocation <= demands + 1e-5), allocation
    assert allocation.sum() <= capacity + 1e-5, allocation
    assert allocation.sum() > 0, allocation
    print(f"{method.__name__}: {np.round(allocation, 3)}; total={allocation.sum():.3f}")

for allocation in (allocate_equal(capacity, demands), allocate_priority(capacity, demands, weights)):
    assert np.all(allocation >= -1e-6)
    assert np.all(allocation <= demands + 1e-5)
    assert allocation.sum() <= capacity + 1e-5
    print(f"baseline: {np.round(allocation, 3)}; total={allocation.sum():.3f}")

assert 0 <= fairness_index(allocate_nash(capacity, demands, weights)) <= 1
assert np.allclose(allocate_nash(200, demands, weights), demands)
assert np.allclose(allocate_nash(0, demands, weights), np.zeros(3))
print("All basic correctness checks passed.")
