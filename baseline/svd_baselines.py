import numpy as np

def svd_baseline(X_trunc, dim):
    U, S, Vt = np.linalg.svd(X_trunc, full_matrices=False)
    vectors = U[:, :dim] * S[:dim]
    return vectors

def svd_s_baseline(X_trunc, dim):
    X_sqrt = np.sqrt(np.maximum(X_trunc, 0))
    U, S, Vt = np.linalg.svd(X_sqrt, full_matrices=False)
    vectors = U[:, :dim] * S[:dim]
    return vectors

def svd_l_baseline(X_trunc, dim):
    X_log = np.log1p(X_trunc)
    U, S, Vt = np.linalg.svd(X_log, full_matrices=False)
    vectors = U[:, :dim] * S[:dim]
    return vectors
