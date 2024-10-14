import numpy as np
import config

def weight_func(x, x_max=config.X_MAX, alpha=config.ALPHA):
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x, dtype=np.float64)
        mask = x < x_max
        result[mask] = (x[mask] / x_max) ** alpha
        result[~mask] = 1.0
        return result
    if x < x_max:
        return (x / x_max) ** alpha
    return 1.0
