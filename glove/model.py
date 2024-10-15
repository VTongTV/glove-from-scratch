import numpy as np
import config
from glove.weighting import weight_func

class GloVeModel:
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W = None
        self.W_tilde = None
        self.b = None
        self.b_tilde = None
        self._init_params()

    def _init_params(self):
        rng = np.random.RandomState(config.SEED)
        scale = 0.5 / self.embedding_dim
        self.W = (rng.rand(self.vocab_size, self.embedding_dim) - 0.5) * 2 * scale
        self.W_tilde = (rng.rand(self.vocab_size, self.embedding_dim) - 0.5) * 2 * scale
        self.b = np.zeros(self.vocab_size)
        self.b_tilde = np.zeros(self.vocab_size)

    def compute_cost(self, rows, cols, vals):
        diff = np.sum(self.W[rows] * self.W_tilde[cols], axis=1)
        diff += self.b[rows] + self.b_tilde[cols] - np.log(vals)
        f_x = weight_func(vals)
        return np.sum(f_x * diff ** 2)

    def compute_grad_w(self, i, j, x_ij):
        diff = np.dot(self.W[i], self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(x_ij)
        f_x = weight_func(x_ij)
        grad_w_i = f_x * diff * self.W_tilde[j]
        grad_w_tilde_j = f_x * diff * self.W[i]
        return grad_w_i, grad_w_tilde_j
