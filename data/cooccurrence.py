import numpy as np
from collections import defaultdict
import config

class CooccurrenceMatrix:
    def __init__(self, vocab, window_size=config.WINDOW_SIZE, context_type=config.CONTEXT_TYPE):
        self.vocab = vocab
        self.window_size = window_size
        self.context_type = context_type
        self.entries = defaultdict(lambda: defaultdict(float))

    def build(self, token_stream):
        self.entries = defaultdict(lambda: defaultdict(float))
        window = []
        for tokens in token_stream:
            for token in tokens:
                if token not in self.vocab:
                    continue
                window.append(token)
                if len(window) > 2 * self.window_size + 1:
                    window.pop(0)
                self._update_window(window)

    def _update_window(self, window):
        raise NotImplementedError

    def nonzero_entries(self):
        for i_word, contexts in self.entries.items():
            for j_word, count in contexts.items():
                if count > 0:
                    yield self.vocab.word2idx[i_word], self.vocab.word2idx[j_word], count

    def to_triplets(self):
        rows, cols, vals = [], [], []
        for i_word, contexts in self.entries.items():
            for j_word, count in contexts.items():
                if count > 0:
                    rows.append(self.vocab.word2idx[i_word])
                    cols.append(self.vocab.word2idx[j_word])
                    vals.append(count)
        return np.array(rows), np.array(cols), np.array(vals)
