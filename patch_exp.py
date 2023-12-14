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
        self.save_dir = save_dir if save_dir is not None else "outputs/"
        self.reset_data()
    
    def reset_data(self) -> None:
        self.data_engine = None
        self.data = None
        self.n_patches = None
        self.patch_size = None
        self.ksvd = None
    
    def generate_train_patches(
            self, 
            dataset_name: str, 
            n_patches: int, 
            patch_size: int, 
            return_data: bool = False
        ) -> None | np.ndarray:
        """
        DESCRIPTION:
            Generate patches data from a given dataset.
        ARGS:
            - dataset_name: name of the dataset to use. Must be one of those defined in PatchDataGenerator.DATALOADERS keys.
            - n_patches: number of patches to generate
            - patch_size: size of the patches
            - return_data: if True, return the generated patches
        """
        
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
        """
        DESCRIPTION:
            Train a dictionary from scratch using KSVD method on self.data signals dataset.
        ARGS:
            [inherited from train_dictionary. See its docstring for details]
        """
        
        if self.data is None:
            raise ValueError("You must generate the training patches before training a K-SVD dictionary from scratch.")

        # run KSVD
        ksvd = KSVD(
            n_atoms=n_atoms,
            sparsity=sparsity,
            pursuit_method=pursuit_method,
            use_dc_atom=True,
            dict_name=dict_name,
            save_dir=self.save_dir,
            verbose=verbose,
        )
        print(f"Training KSVD dictionary from scratch for {max_iter} iterations...")
        ksvd.fit(
            y=self.data, 
            max_iter=max_iter, 
            save_chekpoints=save_chekpoints,
            return_reconstruction=False,
        )
        self.ksvd = ksvd
    
    def __train_from_checkpoint(
            self, 
            dict_name: str,
            checkpoint: str,
            sparsity: int,
            max_iter: int,
            pursuit_method: Pursuit,
            save_chekpoints: bool,
            verbose: bool = True,
        ) -> None:
        """
        DESCRIPTION:
            Train a dictionary from a checkpoint using KSVD method on self.data signals dataset.
        ARGS:
            [inherited from train_dictionary. See its docstring for details]
        """
        
        if self.data is None:
            raise ValueError("You must generate the training patches before training a K-SVD dictionary from a checkpoint.")

        # load checkpoint
        if not os.path.isfile(checkpoint):
            raise ValueError(f"The checkpoint file '{checkpoint}' does not exist.")
        checkpoint_dict = np.load(checkpoint)
        checkpoint_dict = checkpoint_dict / np.linalg.norm(checkpoint_dict, axis=0) # normalize atoms

        # run KSVD
        n_atoms = checkpoint_dict.shape[1]
        ksvd = KSVD(
            n_atoms=n_atoms, 
            sparsity=sparsity, 
            pursuit_method=pursuit_method, 
            use_dc_atom=True,
            init_dict=checkpoint_dict,
            dict_name=dict_name,
            save_dir=self.save_dir,
            verbose=verbose,
        )
        print(f"Training KSVD dictionary from checkpoint [{checkpoint}] for {max_iter} iterations...")
        ksvd.fit(
            y=self.data, 
            max_iter=max_iter, 
            save_chekpoints=save_chekpoints,
            return_reconstruction=False,
        )
        self.ksvd = ksvd
    
    def train_dictionary(
            self,
            dict_name: str,
            sparsity: int,
            max_iter: int,
            pursuit_method: Pursuit,
            n_atoms: int = None,
            checkpoint: str = None,
            save_chekpoints: bool = True,
            return_dict: bool = False,
            save: bool = True,
            verbose: bool = True,
        ) -> None:
        """
        DESCRIPTION:
            Train a dictionary using KSVD method on a given signals dataset.
        ARGS:
            - dict_name: name given to the dictionary to train
            - sparsity: max number of atoms from the dictionary to represent each signal
            - max_iter: max number of iterations in the main loop of KSVD
            - pursuit_method: pursuit method to use in KSVD
            - n_atoms: number of atoms in the the dictionary to train
            - checkpoint: path to a dictionary checkpoint to init it before training
            - save_chekpoints: if True, save checkpoints of the dictionary during training
            - return_dict: if True, return the trained dictionary
            - save: if True, save the trained dictionary and residual history
        """
        
        if n_atoms is None and checkpoint is None:
            raise ValueError("You must specify either the number of atoms or a checkpoint to train a dictionary.")
        
        if n_atoms is not None and checkpoint is not None:
            raise ValueError("You can't specify both the number of atoms and a checkpoint to train a dictionary.")

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
        
        elif checkpoint is not None:
            self.__train_from_checkpoint(
                dict_name=dict_name,
                checkpoint=checkpoint,
                sparsity=sparsity,
                max_iter=max_iter,
                pursuit_method=pursuit_method,
                save_chekpoints=save_chekpoints,
                verbose=verbose,
            )

        atom_means = np.mean(self.ksvd.dict, axis=0)
        if np.max(atom_means[1:]) > 1e-8: # skip the DC atom
            idx = np.argmax(atom_means[1:]) + 1
            print(f"WARNING: Atom {idx} has an excessive mean value = {atom_means[idx]}")
        
        if save:
            save_dict_dir = os.path.join(self.save_dir, dict_name)
            Path(save_dict_dir).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(save_dict_dir, f"{dict_name}.npy"), self.ksvd.dict)
            np.save(os.path.join(save_dict_dir, f"{dict_name}_res.npy"), np.array(self.ksvd.residual_history))
        
        if return_dict:
            return self.ksvd.dict


def main():

    # CHOOSE DATASET TO LEARN THE KSVD DICTIONARY
    dataset_name = "olivetti"

    save_dir = "outputs/"
    dict_name = f"ksvd_{dataset_name}"

    processor = KSVDProcessor(save_dir=save_dir)

    # GENERATE TRAINNG PATCHES
    processor.generate_train_patches(
        dataset_name=dataset_name,
        n_patches = 11_000,
        patch_size = 8,
        return_data=False,
    )

    # LEARN KSVD DICTIONARY
    ksvd_dict = processor.train_dictionary(
        n_atoms=441, # from scratch
        # checkpoint=os.path.join(save_dir, f"{dict_name}/{dict_name}.npy"), # from checkpoint
        sparsity=10,
        max_iter=50,
        pursuit_method=OrthogonalMatchingPursuit,
        dict_name=dict_name,
        save_chekpoints=True,
        return_dict=True,
    )

    # PLOT LEARNED KSVD DICTIONARY
    ksvd_patch_dict = PatchDictionary(
        dict=ksvd_dict, # dict=np.load(os.path.join(save_dir, f"ksvd_{dataset_name}/ksvd_{dataset_name}.npy")),
        dict_name=dict_name,
        save_dir=os.path.join(save_dir, dict_name),
    )
    ksvd_patch_dict.plot_dictionary(ncol_plot=21, borders=True, save=True) # 21x21 = 441 atoms
    

if __name__ == "__main__":
    main()