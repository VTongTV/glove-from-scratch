import numpy as np
import config

class GloVeModel:
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W = None
        self.W_tilde = None
        self.b = None
        self.b_tilde = None
