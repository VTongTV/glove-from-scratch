import numpy as np
import config

class AdaGrad:
    def __init__(self, vocab_size, embedding_dim, lr=config.INITIAL_LR):
        self.lr = lr
        eps = 1e-8
        self.grad_sq_W = np.full((vocab_size, embedding_dim), eps)
        self.grad_sq_W_tilde = np.full((vocab_size, embedding_dim), eps)
        self.grad_sq_b = np.full(vocab_size, eps)
        self.grad_sq_b_tilde = np.full(vocab_size, eps)

    def update(self, model, i, j, grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j):
        self.grad_sq_W[i] += grad_w_i ** 2
        self.grad_sq_W_tilde[j] += grad_w_tilde_j ** 2
        self.grad_sq_b[i] += grad_b_i ** 2
        self.grad_sq_b_tilde[j] += grad_b_tilde_j ** 2

        model.W[i] -= self.lr * grad_w_i / np.sqrt(self.grad_sq_W[i])
        model.W_tilde[j] -= self.lr * grad_w_tilde_j / np.sqrt(self.grad_sq_W_tilde[j])
        model.b[i] -= self.lr * grad_b_i / np.sqrt(self.grad_sq_b[i])
        model.b_tilde[j] -= self.lr * grad_b_tilde_j / np.sqrt(self.grad_sq_b_tilde[j])
