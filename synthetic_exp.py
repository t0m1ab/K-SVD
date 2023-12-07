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

from functools import partial
from multiprocessing.pool import Pool

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from dictionary_learning import KSVD
from synthetic_data import SyntheticDataGenerator
from utils import plot_results_synthetic_exp


def run_ksvd_synthetic_experiment(
        n_features: int,
        n_signals: int,
        n_atoms: int,
        sparsity: int,
        noise_std: float,
        max_iter: int,
        pursuit_method: Pursuit,
        success_threshold: float,
        logs: bool = False,
    ) -> int:
    """
    Run KSVD experiment on synthetic data and return the success score.
    """

    if logs:
        print(f"Exp PID={os.getpid()} just started")

    # create synthetic data
    data_engine = SyntheticDataGenerator(n_features=n_features)
    data_engine.create_synthetic_dictionary(n_atoms=n_atoms, normalize_columns=True, return_dict=False)
    y = data_engine.create_synthetic_signals(n_signals=n_signals, sparsity=sparsity, noise_std=noise_std, return_signals=True)

    if logs:
        print(f"Exp PID={os.getpid()} created data")

    # run KSVD
    ksvd = KSVD(n_atoms=n_atoms, sparsity=sparsity, pursuit_method=pursuit_method, verbose=False)
    ksvd.fit(y=y, max_iter=max_iter, return_reconstruction=False)

    if logs:
        print(f"Exp PID={os.getpid()} executed KSVD")

    # compute score
    score = data_engine.sucess_score(designed_dict=ksvd.dict, threshold=success_threshold)

    if logs:
        print(f"Exp PID={os.getpid()} finished with score={score}")

    return score


def run_ksvd_synthetic_experiment_wrapper(_, **kwargs) -> int:
    """
    Wrapper for $run_ksvd_synthetic_experiment$ to be used when multiprocessing with Pool.map for example.
    """
    # kwargs["logs"] = True # to log the PID running each experiment
    return run_ksvd_synthetic_experiment(**kwargs)


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
        n_process: int = None,
        plot: bool = False,
        save: bool = False,
        save_dir: str = None,
    ):
    """
    Run KSVD $n_runs$ times with the given parameters and return statistics on the success scores.
    """

    # define noise_std from noise_db considering that generating dictionary values follow U[-a,a]
    noise_std = (10 ** (-float(noise_db)/20)) / np.sqrt(n_features) if noise_db > 0 else 0

    # define number of processes to run the experiments
    n_process = n_process if n_process is not None else os.cpu_count()

    # common parameters for each experiment
    parameters = {
        "n_features": n_features,
        "n_signals": n_signals,
        "n_atoms": n_atoms,
        "sparsity": sparsity,
        "noise_std": noise_std,
        "max_iter": max_iter,
        "pursuit_method": pursuit_method,
        "success_threshold": success_threshold
    }

    if n_process > 1: # parallel
        with Pool(n_process) as p:
            pool_result = list(tqdm(
                p.imap(
                    partial(
                        run_ksvd_synthetic_experiment_wrapper, 
                        **parameters,
                        ), 
                    range(n_runs)
                ), 
                total=n_runs, 
                desc=f"Running experiments for noise {noise_db}dB with {n_process} processes"
            ))
    else: # sequential
        pool_result = []
        for _ in tqdm(range(n_runs), desc=f"Running experiments for noise {noise_db}dB sequentially"):
            pool_result.append(run_ksvd_synthetic_experiment(**parameters))
    
    scores = np.array(pool_result)
    print(f"Average success score = {np.mean(scores)}/{np.std(scores):.2f} (mean/std over {n_runs} runs)")

    # if plot:
    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     ax.scatter(np.zeros(n_runs), scores, marker="s", color="black")
    #     ax.set_title(f"Success scores over {n_runs} runs for the synthetic experiment")
    #     ax.set_xticks([])
    #     ax.set_ylabel(f"success score (threshold={success_threshold})")
    #     if plot:
    #         plt.show()

    if save:
        save_dir = "outputs/" if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_dir, f"success_scores_{noise_db}dB.npy"), scores)
        # fig.savefig(os.path.join(save_dir, f"success_scores_{noise_db}dB.png"))


def main():

    save_dir = "synthetic_experiments/"

    for noise_db in [0, 10, 20, 30]: # 0 means no noise

        mutli_run_ksvd_synthetic_experiment(
            n_features = 20,
            n_signals = 1500,
            n_atoms = 50,
            sparsity = 3,
            noise_db = noise_db,
            max_iter = 80,
            n_runs = 50,
            pursuit_method = OrthogonalMatchingPursuit,
            success_threshold = 0.01,
            n_process=1,
            plot = False,
            save = True,
            save_dir = save_dir,
        )

    plot_results_synthetic_exp(
        dir=save_dir,
        n_runs=50,
        success_threshold=0.01,
        plot_groups=False,
        group_size=10,
    )
    

if __name__ == "__main__":
    main()