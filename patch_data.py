import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import os
from pathlib import Path

from utils import create_haar_dict, create_dct_dict


class PatchDataGenerator:
    
    def __init__(self, dataset_name: str = "olivetti", save_dir: str = None) -> None:
        self.dataset_name = None
        self.data_loader = None
        if dataset_name == "olivetti":
            self.dataset_name = dataset_name
            self.data_loader = lambda : fetch_olivetti_faces(shuffle=True, random_state=42)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        self.patch_size = None
        self.n_patches = None
        self.data = None
        self.save_dir = ""
        if save_dir is not None:
            self.save_dir = save_dir
            Path(save_dir).mkdir(parents=True, exist_ok=True) # default directory to save plots
  
    def create_patch_dataset(self, patch_size: int, n_patches: int, return_data: bool = False) -> None | np.ndarray:
        """ 
        Create a patch dataset of $n_patches$ patches of size $patch_size$ from the dataset $self.dataset_name$.
        """

        if self.data is not None:
            raise ValueError("A patch dataset was already created and is stored in self.data.")

        self.patch_size = patch_size
        self.n_patches = n_patches

        # build all patches
        dataset = self.data_loader()
        n_images = dataset.images.shape[0]
        patches_list = []
        for idx in range(n_images):
            for patch in self.extract_all_patches(dataset.images[idx], patch_size):
                patches_list.append(patch)

        # randomly choose patches
        self.data = np.zeros((patch_size**2, n_patches))
        chosen_patches_idx = np.random.choice(len(patches_list), n_patches, replace=False)
        for patch_idx in range(n_patches):
            self.data[:,patch_idx] = patches_list[chosen_patches_idx[patch_idx]]

        print(f"Patch dataset {self.dataset_name.upper()} was created with {self.n_patches} patches of size {self.patch_size}x{self.patch_size}.")

        if return_data:
            return self.data
    
    def extract_all_patches(self, image: np.ndarray, patch_size: int) -> list:
        """ 
        Extract all patches of size $patch_size$ from the image $image$.
        """
        image_height = image.shape[0]
        image_width = image.shape[1]
        ncol_patches = image_width // patch_size
        nrow_patches = image_height // patch_size

        patches = []
        for row in range(nrow_patches):
            for col in range(ncol_patches):
                up = row * patch_size
                left = col * patch_size
                patches.append(image[up:up+patch_size, left:left+patch_size].reshape(-1))

        return patches

    def plot_random_patches(self, n: int = 16, ncol_plot: int = 4, save: bool = False):
        """
        Plot a random patches from the $self.data$.
        """

        if self.data is None:
            raise ValueError("You must create a patch dataset before plotting a random patch.")
        
        # get random patches
        random_patches = []
        random_indexes = np.random.choice(self.data.shape[1], n, replace=False)
        for idx in random_indexes:
            random_patches.append(self.data[:,idx].reshape((self.patch_size, self.patch_size)))

        # plot random patches
        nrow = (n // ncol_plot) + 1 if (n % ncol_plot) > 0 else (n // ncol_plot)
        fig, axes = plt.subplots(nrow, ncol_plot, figsize=(10, 2*nrow),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.4, wspace=0.1)
        )
        for idx, ax in enumerate(axes.flat):
            if idx >= n:
                break
            ax.imshow(random_patches[idx], cmap='gray')
            ax.set_title(f"Patch {random_indexes[idx]}")
        plt.suptitle(f"Random {self.patch_size}x{self.patch_size} patches from the {self.dataset_name} dataset", size=16)
        if save:
            plt.savefig(os.path.join(self.save_dir, f"random_patches_n={n}.png"), dpi=300)
        else:
            plt.show()
    
    def plot_collection(self, n: int = 500, nrow_plot: int = 10, sort_variance: bool = True, save: bool = False):
        """
        Plot a collection of $n$ patches sorted according to their variance if $sort_variance$ is True.
        """

        if self.data is None:
            raise ValueError("You must create a patch dataset before plotting a collection.")
        ncol_plot = n // nrow_plot

        if ncol_plot * nrow_plot != n:
            n = ncol_plot * nrow_plot
            print(f"Number of patches in the collection was set to {n} to be a multiple of nrow_plot={nrow_plot}.")
        
        # get random patches
        random_indexes = np.random.choice(self.data.shape[1], n, replace=False)
        patches_idx_and_variance = [(idx, np.var(self.data[:,idx])) for idx in random_indexes]
        if sort_variance:
            patches_idx_and_variance.sort(key=lambda x: x[1], reverse=False)

        # building the collection array and position all patches
        collection = np.zeros((nrow_plot*self.patch_size, ncol_plot*self.patch_size))
        for k, (patch_idx, _) in enumerate(patches_idx_and_variance):
            row = (k // ncol_plot) * self.patch_size
            col = (k % ncol_plot) * self.patch_size
            patch = self.data[:,patch_idx].reshape((self.patch_size, self.patch_size))
            collection[row:row+self.patch_size, col:col+self.patch_size] = patch
        
        # plot collection
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(collection, cmap='gray')
        ax.set_title(f"Collection of {n} patches", size=16)
        plt.axis('off')
        fig.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, f"collection_n={n}.png"), dpi=300)
        else:
            plt.show()


class PatchDictionary:

    def __init__(self, dict: np.ndarray, dict_name: str = None, save_dir: str = None) -> None:
        self.dict = dict
        self.n_atoms = dict.shape[1]
        self.patch_size = int(np.sqrt(dict.shape[0]))
        if self.patch_size ** 2 != dict.shape[0]:
            raise ValueError("The dictionary doesn't contain square patches in its columns.")
        self.dict_name = dict_name if dict_name is not None else "Unknown dictionary"
        self.save_dir = ""
        if save_dir is not None:
            self.save_dir = save_dir
            Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    def __plot_dictionary_without_borders(self, ncol_plot: int, transpose_dict: bool, save: bool):

        if ncol_plot is None:
            ncol_plot = int(np.sqrt(self.dict.shape[1]))
            if ncol_plot ** 2 < self.n_atoms: # not a perfect square number of atoms
                ncol_plot += 1
        
        nrow = (self.n_atoms // ncol_plot) + 1 if (self.n_atoms % ncol_plot) > 0 else (self.n_atoms // ncol_plot)

        # build the array containing all patches
        collection = np.ndarray((nrow*self.patch_size, ncol_plot*self.patch_size))
        for idx in range(self.n_atoms):
            row = (idx // ncol_plot) * self.patch_size
            col = (idx % ncol_plot) * self.patch_size
            patch = self.dict[:,idx].reshape((self.patch_size, self.patch_size))
            if not transpose_dict:
                collection[row:row+self.patch_size, col:col+self.patch_size] = patch
            else:
                collection[col:col+self.patch_size, row:row+self.patch_size] = patch
        
        # plot collection
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(collection, cmap='gray')
        ax.set_title(f"{self.dict_name} dictionary with {self.n_atoms} atoms", size=16)
        plt.axis('off')
        fig.tight_layout()
        if save:
            file_name = f"{self.dict_name.lower().replace(' ','-')}_dict.png"
            plt.savefig(os.path.join(self.save_dir, file_name), dpi=300)
        else:
            plt.show()
    
    def __plot_dictionary_with_borders(self, transpose_dict: bool, save: bool):
        """
        Plot dictionary with borders between patches.
        """

        n_patches = self.dict.shape[1]
        n_patches_edge = int(np.sqrt(n_patches))
        if n_patches_edge ** 2 != n_patches:
            raise ValueError("The dictionary must contain a square number of patches.")
        
        # build the array containing all patches
        collection_edge = n_patches_edge * (self.patch_size + 1) - 1 # one line between each patch
        collection_with_borders = np.zeros((collection_edge, collection_edge))
        for patch_idx in range(n_patches):
            # row/col position of the patch in the collection
            row = (patch_idx // n_patches_edge)
            col = (patch_idx % n_patches_edge)
            # pixel position of the patch in the collection
            up = row * self.patch_size
            left = col * self.patch_size
            # extract patch from the dictionary and copy it in the collection with borders
            patch = self.dict[:, patch_idx].reshape((self.patch_size, self.patch_size))
            if not transpose_dict:
                collection_with_borders[up+row:up+row+self.patch_size, left+col:left+col+self.patch_size] = patch
            else:
                collection_with_borders[left+col:left+col+self.patch_size, up+row:up+row+self.patch_size] = patch

        # plot collection with borders
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(collection_with_borders, cmap='gray')
        ax.set_title(f"{self.dict_name} dictionary with {self.n_atoms} atoms", size=16)
        plt.axis('off')
        fig.tight_layout()
        if save:
            file_name = f"{self.dict_name.lower().replace(' ','-')}_dict.png"
            plt.savefig(os.path.join(self.save_dir, file_name), dpi=300)
        else:
            plt.show()
    
    def plot_dictionary(self, ncol_plot: int = None, borders: bool = False, transpose_dict: bool = False, save: bool = False):
        if not borders:
            self.__plot_dictionary_without_borders(ncol_plot=ncol_plot, transpose_dict=transpose_dict, save=save)
        else:
            self.__plot_dictionary_with_borders(transpose_dict=transpose_dict, save=save)


if __name__ == "__main__":
    
    ## Face patch dataset
    # data_engine = PatchDataGenerator(dataset_name="olivetti", save_dir="patch_experiments/")
    # patches = data_engine.create_patch_dataset(patch_size=8, n_patches=1000, return_data=False)
    # data_engine.plot_random_patches(save=True)
    # data_engine.plot_collection(n=500, nrow_plot=10, sort_variance=True, save=True)

    ## Haar basis (don't normalize atoms for visualization purpose)
    haar_dict = PatchDictionary(
        dict=create_haar_dict(patch_size=8, K=441, normalize_atoms=False, transpose_dict=False),
        dict_name="Haar",
        save_dir="patch_experiments/"
    )
    haar_dict.plot_dictionary(borders=True, save=True)

    ## DCT basis (don't normalize atoms for visualization purpose)
    dct_dict = PatchDictionary(
        dict=create_dct_dict(patch_size=8, K=441, normalize_atoms=False, transpose_dict=True),
        dict_name="DCT",
        save_dir="patch_experiments/"
    )
    dct_dict.plot_dictionary(borders=True, save=True)
