import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from glove.weighting import weight_func

def test_weight_func_zero():
    assert weight_func(0) == 0.0

def test_weight_func_at_x_max():
    assert weight_func(100) == 1.0

def test_weight_func_above_x_max():
    assert weight_func(200) == 1.0

def test_weight_func_below_x_max():
    val = weight_func(50, x_max=100, alpha=0.75)
    expected = (50.0 / 100.0) ** 0.75
    assert abs(val - expected) < 1e-9

def test_weight_func_non_decreasing():
    vals = [weight_func(x) for x in range(0, 201, 10)]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1]

def test_weight_func_vectorized():
    x = np.array([0, 50, 100, 200])
    result = weight_func(x)
    assert result[0] == 0.0
    assert abs(result[1] - (0.5 ** 0.75)) < 1e-9
    assert result[2] == 1.0
    assert result[3] == 1.0
