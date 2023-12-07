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

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from dictionary_learning import KSVD
from patch_data import PatchDataGenerator, PatchBasis


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
    ksvd = KSVD(n_atoms=n_atoms, sparsity=sparsity, pursuit_method=pursuit_method, use_dc_atom=True, verbose=True)
    ksvd.fit(y=data, max_iter=max_iter, return_reconstruction=False)

    if save:
        save_dir = save_dir if save_dir is not None else ""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_dir, f"patch_dict_K={n_atoms}.npy"), ksvd.dict)
        np.save(os.path.join(save_dir, f"residual_hist_K={n_atoms}.npy"), np.array(ksvd.residual_history))
    
    if return_dict:
        return ksvd.dict


def main():

    save_dir = "patch_experiments/"

    run_ksvd_image_experiment(
        patch_size = 8,
        n_patches = 11_000,
        n_atoms = 441,
        sparsity = 10,
        max_iter = 50,
        pursuit_method = OrthogonalMatchingPursuit,
        save = True,
        save_dir = save_dir,
        return_dict=False,
    )

    basis = PatchBasis(
        dict = np.load(os.path.join(save_dir, f"patch_dict_K=441.npy")),
        basis_name = "KSVD",
        save_dir = save_dir,
    )

    basis.plot_dictionary(ncol_plot=21, save=True) # 21x21 = 441 atoms
    

if __name__ == "__main__":
    main()