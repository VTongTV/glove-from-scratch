import numpy as np
import config
from glove.model import GloVeModel
from glove.adagrad import AdaGrad
import logging

logger = logging.getLogger(__name__)

def prepare_training_data(cooccurrence):
    rows, cols, vals = cooccurrence.to_triplets()
    return rows, cols, vals

def shuffle_data(rows, cols, vals, seed=config.SEED):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(vals))
    return rows[indices], cols[indices], vals[indices]

def train_step(model, optimizer, i, j, x_ij):
    grad_w_i, grad_w_tilde_j = model.compute_grad_w(i, j, x_ij)
    grad_b_i, grad_b_tilde_j = model.compute_grad_b(i, j, x_ij)
    optimizer.update(model, i, j, grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j)
    diff = np.dot(model.W[i], model.W_tilde[j]) + model.b[i] + model.b_tilde[j] - np.log(x_ij)
    from glove.weighting import weight_func
    f_x = weight_func(x_ij)
    return f_x * diff ** 2
