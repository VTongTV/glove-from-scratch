import numpy as np

def rank_data(x):
    arr = np.array(x, dtype=np.float64)
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[order[j]] == arr[order[j + 1]]:
            j += 1
        if j > i:
            avg_rank = np.mean(ranks[order[i:j + 1]])
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        i = j + 1
    return ranks

def spearman_rho(x, y):
    rx = rank_data(x)
    ry = rank_data(y)
    n = len(x)
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))

def evaluate_similarity(vectors, pairs):
    model_scores = []
    human_scores = []
    for w1, w2, score in pairs:
        if w1 in vectors and w2 in vectors:
            from eval.similarity import cosine_similarity
            cs = cosine_similarity(vectors[w1], vectors[w2])
            model_scores.append(cs)
            human_scores.append(score)
    if len(model_scores) < 2:
        return 0.0
    return spearman_rho(model_scores, human_scores)
