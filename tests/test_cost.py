import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from glove.model import GloVeModel

def test_cost_on_known_input():
    model = GloVeModel(vocab_size=3, embedding_dim=2)
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 0])
    vals = np.array([10.0, 5.0, 2.0])
    cost = model.compute_cost(rows, cols, vals)
    assert cost > 0

def test_cost_decreases_after_step():
    model = GloVeModel(vocab_size=3, embedding_dim=2)
    from glove.adagrad import AdaGrad
    optimizer = AdaGrad(3, 2)
    rows = np.array([0, 1])
    cols = np.array([1, 0])
    vals = np.array([10.0, 5.0])
    cost_before = model.compute_cost(rows, cols, vals)
    for _ in range(10):
        grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j, _ = model.compute_gradients(0, 1, 10.0)
        optimizer.update(model, 0, 1, grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j)
    cost_after = model.compute_cost(rows, cols, vals)
    assert cost_after < cost_before

def test_gradient_numerical_check():
    model = GloVeModel(vocab_size=2, embedding_dim=3)
    i, j, x_ij = 0, 1, 5.0
    grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j, _ = model.compute_gradients(i, j, x_ij)

    eps = 1e-5
    for d in range(3):
        model.W[i, d] += eps
        cost_plus = _single_cost(model, i, j, x_ij)
        model.W[i, d] -= 2 * eps
        cost_minus = _single_cost(model, i, j, x_ij)
        model.W[i, d] += eps
        numerical = (cost_plus - cost_minus) / (2 * eps)
        assert abs(grad_w_i[d] - numerical) / (abs(grad_w_i[d]) + abs(numerical) + 1e-8) < 1e-4

def _single_cost(model, i, j, x_ij):
    from glove.weighting import weight_func
    diff = np.dot(model.W[i], model.W_tilde[j]) + model.b[i] + model.b_tilde[j] - np.log(x_ij)
    return weight_func(x_ij) * diff ** 2
