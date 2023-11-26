""" 
This file runs the experiments performed in the following paper:
K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation
Michal Aharon - Michael Elad - Alfred Bruckstein
IEEE Transactions on Signal Processing, 2006
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from dictionary_learning import KSVD
from synthetic_data import SyntheticData
from utils import plot_results_synthetic_exp


def mutli_run_ksvd_synthetic_experiment(
        n_features: int = 20,
        n_signals: int = 1500,
        n_atoms: int = 50,
        sparsity: int = 3,
        noise_db: float = 0,
        max_iter: int = 80,
        n_runs: int = 50,
        pursuit_method: Pursuit = OrthogonalMatchingPursuit,
        success_threshold: float = 0.01,
        plot: bool = False,
        save: bool = False,
        save_dir: str = None,
    ):
    """
    Run KSVD $n_runs$ times with the given parameters and return statistics on the success scores.
    """

    noise_std = 10 ** (-float(noise_db)/10) if noise_db > 0 else 0
    scores = np.zeros(n_runs)
    for run_idx in tqdm(range(n_runs), desc=f"Running experiments with noise {noise_db}dB"):

        # create synthetic data
        data = SyntheticData(n_features=n_features)
        data.create_synthetic_dictionary(n_atoms=n_atoms, normalize_columns=True, return_dict=False)
        y = data.create_synthetic_signals(n_signals=n_signals, sparsity=sparsity, noise_std=noise_std, return_signals=True)

        # run KSVD
        ksvd = KSVD(n_atoms=n_atoms, sparsity=sparsity, pursuit_method=pursuit_method, verbose=False)
        ksvd.fit(y=y, max_iter=max_iter, return_reconstruction=False)

        # compute score
        scores[run_idx] = data.sucess_score(designed_dict=ksvd.dict, threshold=success_threshold)

    print(f"Average success score = {np.mean(scores)}/{np.std(scores):.2f} (mean/std over {n_runs} runs)")

    if save or plot:

        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.scatter(np.zeros(n_runs), scores, marker="s", color="black")
        # ax.set_title(f"Success scores over {n_runs} runs for the synthetic experiment")
        # ax.set_xticks([])
        # ax.set_ylabel(f"success score (threshold={success_threshold})")

        # if plot:
        #     plt.show()
        if save:
            save_dir = "outputs/" if save_dir is None else save_dir
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(save_dir, f"success_scores_{noise_db}dB.npy"), scores)
            # fig.savefig(os.path.join(save_dir, f"success_scores_{noise_db}dB.png"))


def main():

    save_dir = "synthetic_experiments/"

    for noise_db in [0, 10, 20, 30]:

        mutli_run_ksvd_synthetic_experiment(
            n_features = 20,
            n_signals = 1500,
            n_atoms = 50,
            sparsity = 3,
            noise_db = noise_db,
            max_iter = 2,
            n_runs = 5,
            pursuit_method = OrthogonalMatchingPursuit,
            success_threshold = 0.01,
            plot = False,
            save = True,
            save_dir = save_dir,
        )

    plot_results_synthetic_exp(
        dir=save_dir,
        n_runs=50,
        success_threshold=0.01
    )
    

if __name__ == "__main__":
    main()