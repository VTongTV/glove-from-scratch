import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(vectors, words, output_path):
    vecs = np.array([vectors[w] for w in words if w in vectors])
    labels = [w for w in words if w in vectors]
    if len(vecs) < 2:
        return
    proj = TSNE(n_components=2, random_state=42).fit_transform(vecs)
    plt.figure(figsize=(12, 12))
    plt.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.5)
    for i, label in enumerate(labels):
        plt.annotate(label, (proj[i, 0], proj[i, 1]), fontsize=6)
    plt.savefig(output_path)
    plt.close()
