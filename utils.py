import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_results_synthetic_exp(dir: str = None, n_runs: int = 50, success_threshold: float = 0.01):
    """ Plot the results of the synthetic experiment over different noise levels. """
    
    noise_levels = [0, 10, 20, 30]
    fig, ax = plt.subplots(figsize=(10, 5))

    for noise_db in noise_levels:
        file_path = os.path.join(dir, f"success_scores_{noise_db}dB.npy")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        scores = np.load(file_path)
        absc = noise_db * np.ones(scores.shape[0])
        ax.scatter(absc, scores, marker="s", color="black")
    
    ax.set_title(f"Success scores over {n_runs} runs for the synthetic experiment", size=16)
    ax.set_xticks(noise_levels, [f"{x}dB" for x in noise_levels])
    ax.set_yticks([25, 30, 35, 40, 45, 50])
    ax.set_xlabel("noise level (dB)")
    ax.set_ylabel(f"success score (threshold={success_threshold})")
    ax.grid(True)
    ax.set_axisbelow(True) # put grid behind the plot
    fig.savefig(os.path.join(dir, "success_scores.png"), dpi=300)
    print(f"Figure 'success_scores.png' saved in '{dir}'!")


if __name__ == "__main__":
    
    plot_results_synthetic_exp(
        dir="synthetic_experiments/",
        n_runs=50,
        success_threshold=0.01
    )