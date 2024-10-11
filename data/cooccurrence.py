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
        all_tokens = []
        for tokens in token_stream:
            all_tokens.extend(t for t in tokens if t in self.vocab)
        for i, target in enumerate(all_tokens):
            start = max(0, i - self.window_size)
            end = min(len(all_tokens), i + self.window_size + 1)
            if self.context_type == "symmetric":
                context_range = range(start, end)
            else:
                context_range = range(start, i)
            for j in context_range:
                if j == i:
                    continue
                context = all_tokens[j]
                distance = abs(j - i)
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

    def save(self, path):
        rows, cols, vals = self.to_triplets()
        np.savez(path, rows=rows, cols=cols, vals=vals)

    def load(self, path):
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        self.entries = defaultdict(lambda: defaultdict(float))
        for i, j, v in zip(data["rows"], data["cols"], data["vals"]):
            i_word = self.vocab.idx2word[int(i)]
            j_word = self.vocab.idx2word[int(j)]
            self.entries[i_word][j_word] = v
