import numpy as np

def save_vectors(path, vectors):
    with open(path, "w", encoding="utf-8") as f:
        words = list(vectors.keys())
        dim = len(vectors[words[0]])
        f.write(f"{len(words)} {dim}\n")
        for word in words:
            vec_str = " ".join(f"{v:.6f}" for v in vectors[word])
            f.write(f"{word} {vec_str}\n")

def load_vectors(path):
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        vocab_size, dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            vectors[word] = vec
    return vectors
