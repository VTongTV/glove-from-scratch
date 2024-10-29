import matplotlib.pyplot as plt
import numpy as np
from eval.similarity import cosine_similarity

def plot_similarity_scatter(vectors, pairs, title, output_path):
    model_scores = []
    human_scores = []
    for w1, w2, score in pairs:
        if w1 in vectors and w2 in vectors:
            model_scores.append(cosine_similarity(vectors[w1], vectors[w2]))
            human_scores.append(score)
    plt.scatter(human_scores, model_scores, s=8, alpha=0.5)
    plt.xlabel("human")
    plt.ylabel("model")
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
