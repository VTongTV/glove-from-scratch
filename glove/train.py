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
    grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j, loss = model.compute_gradients(i, j, x_ij)
    optimizer.update(model, i, j, grad_w_i, grad_w_tilde_j, grad_b_i, grad_b_tilde_j)
    return loss

def train(model, rows, cols, vals, num_iterations=None):
    if num_iterations is None:
        num_iterations = config.ITERATIONS_LARGE if model.embedding_dim >= 300 else config.ITERATIONS_SMALL
    optimizer = AdaGrad(model.vocab_size, model.embedding_dim)
    loss_history = []
    for iteration in range(num_iterations):
        r, c, v = shuffle_data(rows, cols, vals, seed=config.SEED + iteration)
        total_loss = 0.0
        for idx in range(len(v)):
            loss = train_step(model, optimizer, r[idx], c[idx], v[idx])
            total_loss += loss
        avg_loss = total_loss / len(v)
        loss_history.append(avg_loss)
        logger.info(f"iteration {iteration + 1}/{num_iterations} loss={avg_loss:.6f}")
        if len(loss_history) > 1 and abs(loss_history[-2] - avg_loss) / (abs(loss_history[-2]) + 1e-10) < 1e-4:
            logger.info(f"converged at iteration {iteration + 1}")
            break
    return loss_history

if __name__ == "__main__":
    import argparse
    from data.tokenizer import tokenize_file
    from data.vocabulary import Vocabulary
    from data.cooccurrence import CooccurrenceMatrix
    from glove.io import save_vectors

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dim", type=int, default=config.EMBEDDING_DIM)
    parser.add_argument("--window", type=int, default=config.WINDOW_SIZE)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output", default="outputs/vectors.txt")
    args = parser.parse_args()

    vocab = Vocabulary()
    vocab.count_from_file(args.corpus, tokenize_file)
    vocab.filter_top_n(config.VOCAB_SIZE)

    cooc = CooccurrenceMatrix(vocab, window_size=args.window)
    cooc.build(tokenize_file(args.corpus))

    rows, cols, vals = prepare_training_data(cooc)
    model = GloVeModel(len(vocab), embedding_dim=args.dim)
    train(model, rows, cols, vals, num_iterations=args.iterations)

    vectors = model.get_vectors(vocab)
    save_vectors(args.output, vectors)
