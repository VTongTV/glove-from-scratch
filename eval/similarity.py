import numpy as np

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def load_ws353(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    return pairs

def load_mc(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
                except ValueError:
                    continue
    return pairs

def load_rg(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
                except ValueError:
                    continue
    return pairs

def load_scws(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                try:
                    pairs.append((parts[1].lower(), parts[3].lower(), float(parts[7])))
                except ValueError:
                    continue
    return pairs

def load_rw(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
                except ValueError:
                    continue
    return pairs
