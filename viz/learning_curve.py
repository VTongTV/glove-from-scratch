import json
import matplotlib.pyplot as plt

def plot_learning_curve(results_path, output_path):
    with open(results_path) as f:
        data = json.load(f)
    loss = data["loss"]
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    if len(loss) > 1:
        for i in range(1, len(loss)):
            if abs(loss[i] - loss[i - 1]) / (abs(loss[i - 1]) + 1e-10) < 1e-4:
                plt.axvline(x=i + 1, color="r", linestyle="--", label=f"converged at {i + 1}")
                break
    plt.legend()
    plt.savefig(output_path)
    plt.close()
