import matplotlib.pyplot as plt
import json

def plot_analogy_bars(results_path, output_path):
    with open(results_path) as f:
        data = json.load(f)
    models = []
    sem, syn, tot = [], [], []
    for name, acc in data.items():
        models.append(name)
        sem.append(acc.get("semantic_total", {}).get("accuracy", 0) * 100)
        syn.append(acc.get("syntactic_total", {}).get("accuracy", 0) * 100)
        tot.append(acc.get("overall", {}).get("accuracy", 0) * 100)
    x = range(len(models))
    w = 0.25
    plt.bar([i - w for i in x], sem, w, label="semantic")
    plt.bar(x, syn, w, label="syntactic")
    plt.bar([i + w for i in x], tot, w, label="total")
    plt.xticks(x, models)
    plt.ylabel("accuracy %")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
