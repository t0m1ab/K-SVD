""" 
This file runs the image processing experiments performed in the following paper:
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
from patch_data import PatchDataGenerator


def run_ksvd_image_experiment(
        patch_size: int,
        n_patches: int,
        n_atoms: int,
        sparsity: int,
        max_iter: int,
        pursuit_method: Pursuit,
        save: bool = False,
        save_dir: str = None,
        return_dict: bool = False,
    ) -> int:
    """
    Run KSVD experiment on face image data and return the success score.
    """

    # create synthetic data
    data_engine = PatchDataGenerator(dataset_name="olivetti")
    data = data_engine.create_patch_dataset(n_patches=n_patches, patch_size=patch_size, return_data=True)

    # run KSVD
    ksvd = KSVD(n_atoms=n_atoms, sparsity=sparsity, pursuit_method=pursuit_method, use_dc_atom=True, verbose=False)
    ksvd.fit(y=data, max_iter=max_iter, return_reconstruction=False, verbose=True)

    print(ksvd.dict.shape)

    if save:
        save_dir = save_dir if save_dir is not None else ""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_dir, f"patch_dict_K={n_atoms}.npy", ksvd.dict))
    
    if return_dict:
        return ksvd.dict


def main():

    save_dir = "patch_experiments/"

    run_ksvd_image_experiment(
        patch_size = 8,
        n_patches = 11_000,
        n_atoms = 441,
        sparsity = 10,
        max_iter = 20,
        pursuit_method = OrthogonalMatchingPursuit,
        save = True,
        save_dir = save_dir,
        return_dict=False,
    )

    # plot_results_synthetic_exp(
    #     dir=save_dir,
    #     n_runs=50,
    #     success_threshold=0.01,
    #     plot_groups=False,
    #     group_size=10,
    # )
    

if __name__ == "__main__":
    main()