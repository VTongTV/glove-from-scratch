import collections

class Vocabulary:
    def __init__(self):
        self.word2count = collections.Counter()
        self.word2idx = {}
        self.idx2word = {}
        self.total_tokens = 0

    def count(self, tokens):
        self.word2count.update(tokens)
        self.total_tokens += len(tokens)

    def count_from_file(self, path, tokenize_fn):
        for tokens in tokenize_fn(path):
            self.count(tokens)

    def build_indices(self, max_words=None):
        most_common = self.word2count.most_common(max_words)
        self.word2idx = {}
        self.idx2word = {}
        for idx, (word, _) in enumerate(most_common):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def filter_top_n(self, n):
        most_common = self.word2count.most_common(n)
        self.word2count = collections.Counter(dict(most_common))
        self.build_indices()

    def __contains__(self, word):
        return word in self.word2idx

    def __len__(self):
        return len(self.word2idx)
