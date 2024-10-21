import numpy as np

def build_truncated(cooccurrence, vocab, top_n=10000):
    top_indices = list(range(min(top_n, len(vocab))))
    rows, cols, vals = cooccurrence.to_triplets()
    mask = np.isin(cols, top_indices)
    filtered_rows = rows[mask]
    filtered_cols = cols[mask]
    filtered_vals = vals[mask]
    n_rows = len(vocab)
    n_cols = min(top_n, len(vocab))
    X_trunc = np.zeros((n_rows, n_cols))
    for r, c, v in zip(filtered_rows, filtered_cols, filtered_vals):
        if c < n_cols:
            X_trunc[r, c] = v
    return X_trunc
