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
from patch_data import PatchDataGenerator, PatchDictionary


class KSVDProcessor():

    def __init__(self, save_dir: str = None) -> None:
        self.save_dir = save_dir if save_dir is not None else "patch_experiments/"
        self.data_engine = None
        self.data = None
        self.n_patches = None
        self.patch_size = None
        self.ksvd = None
    
    def generate_train_patches(self, dataset_name: str, n_patches: int, patch_size: int, return_data: bool = False) -> None | np.ndarray:
        
        # define data engine from dataset_name
        self.data_engine = PatchDataGenerator(dataset_name=dataset_name)

        # generate patches from the dataset
        self.data = self.data_engine.create_patch_dataset(n_patches=n_patches, patch_size=patch_size, return_data=True)
        self.n_patches = n_patches
        self.patch_size = patch_size

        if return_data:
            return self.data
    
    def __train_from_scratch(
        self, 
        dict_name: str,
        n_atoms: int,
        sparsity: int, 
        max_iter: int,
        pursuit_method: Pursuit,
        save_chekpoints: bool,
        verbose: bool = True,
    ) -> None:
        
        if self.data is None:
            raise ValueError("You must generate the training patches before training a K-SVD dictionary from scratch.")

        # run KSVD
        ksvd = KSVD(
            n_atoms=n_atoms,
            sparsity=sparsity,
            pursuit_method=pursuit_method,
            use_dc_atom=True,
            verbose=verbose
        )
        print(f"Training KSVD dictionary from scratch for {max_iter} iterations...")
        ksvd.fit(
            y=self.data, 
            max_iter=max_iter, 
            return_reconstruction=False,
            dict_name=dict_name,
            path=self.save_dir,
            save_chekpoints=save_chekpoints,
        )
        self.ksvd = ksvd
    
    def __train_from_checkpoint(
        self, 
        dict_name: str,
        checkpoint_path: str,
        sparsity: int,
        max_iter: int,
        pursuit_method: Pursuit,
        save_chekpoints: bool,
        verbose: bool = True,
    ) -> None:
        
        if self.data is None:
            raise ValueError("You must generate the training patches before training a K-SVD dictionary from a checkpoint.")

        # load checkpoint
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"The checkpoint file '{checkpoint_path}' does not exist.")
        checkpoint_dict = np.load(checkpoint_path)

        # run KSVD
        n_atoms = checkpoint_dict.shape[1]
        ksvd = KSVD(
            dict=checkpoint_dict,
            n_atoms=n_atoms, 
            sparsity=sparsity, 
            pursuit_method=pursuit_method, 
            use_dc_atom=True, 
            verbose=verbose
        )
        print(f"Training KSVD dictionary from checkpoint [{checkpoint_path}] for {max_iter} iterations...")
        ksvd.fit(
            y=self.data, 
            max_iter=max_iter, 
            return_reconstruction=False,
            dict_name=dict_name,
            path=self.save_dir,
            save_chekpoints=save_chekpoints,
        )
        self.ksvd = ksvd
    
    def train_dictionary(self,
        dict_name: str,
        sparsity: int,
        max_iter: int,
        pursuit_method: Pursuit,
        save_chekpoints: bool = True,
        n_atoms: int = None,
        checkpoint_path: str = None,
        verbose: bool = True,
        return_dict: bool = False,
        save: bool = True,
    ) -> None:
        
        if n_atoms is None and checkpoint_path is None:
            raise ValueError("You must specify either the number of atoms or a checkpoint to train a dictionary.")
        
        if n_atoms is not None and checkpoint_path is not None:
            raise ValueError("You can't specify both the number of atoms and a checkpoint to train a dictionary")

        if n_atoms is not None:
            self.__train_from_scratch(
                dict_name=dict_name,
                n_atoms=n_atoms,
                sparsity=sparsity,
                max_iter=max_iter,
                pursuit_method=pursuit_method,
                save_chekpoints=save_chekpoints,
                verbose=verbose,
            )
        
        elif checkpoint_path is not None:
            self.__train_from_checkpoint(
                dict_name=dict_name,
                checkpoint_path=checkpoint_path,
                sparsity=sparsity,
                max_iter=max_iter,
                pursuit_method=pursuit_method,
                save_chekpoints=save_chekpoints,
                verbose=verbose,
            )
        
        if save:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(self.save_dir, f"patch_dict_K={n_atoms}.npy"), self.ksvd.dict)
            np.save(os.path.join(self.save_dir, f"residual_hist_K={n_atoms}.npy"), np.array(self.ksvd.residual_history))
        
        if return_dict:
            return self.ksvd.dict


def main():

    save_dir="patch_experiments/"

    processor = KSVDProcessor(save_dir=save_dir)

    # generate training patches
    processor.generate_train_patches(
        dataset_name="olivetti",
        n_patches = 11_000,
        patch_size = 8,
        return_data=False,
    )

    # learn KSVD dictionary
    processor.train_dictionary(
        n_atoms=441, # from scratch
        # checkpoint_path=os.path.join(save_dir, "ksvd_olivetti/ksvd_olivetti_iter=30.npy"), # from checkpoint
        sparsity=10,
        max_iter=50,
        pursuit_method=OrthogonalMatchingPursuit,
        dict_name="ksvd_olivetti",
        save_chekpoints=True,
    )

    # plot KSVD dictionary
    basis = PatchDictionary(
        # dict = np.load(os.path.join(save_dir, "ksvd_olivetti/ksvd_olivetti_iter=30.npy")),
        dict_name = "KSVD",
        save_dir = save_dir,
    )
    basis.plot_dictionary(ncol_plot=21, save=True) # 21x21 = 441 atoms
    

if __name__ == "__main__":
    main()