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
        if len(window) < 2:
            return
        center = len(window) // 2
        target = window[center]
        if target not in self.vocab:
            return
        start = max(0, center - self.window_size)
        end = min(len(window), center + self.window_size + 1)
        for j in range(start, end):
            if j == center:
                continue
            context = window[j]
            if context not in self.vocab:
                continue
            if self.context_type == "asymmetric" and j > center:
                continue
            distance = abs(j - center)
            self.entries[target][context] += 1.0 / distance

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
