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
