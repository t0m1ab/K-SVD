import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import os
from pathlib import Path
from time import time
from datasets import load_dataset
from collections import defaultdict

from utils import create_haar_dict, create_dct_dict


def extract_all_patches(image: np.ndarray, patch_size: int) -> list:
    """ 
    ARGS:
        - image: image from which to extract the patches.
        - patch_size: size of the patches to extract.
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


def load_olivetti_data() -> np.ndarray:
    """ 
    Load Olivetti 64x64 images from Sklearn datasets.
    Returns images as vectors with values in [0,1].
    Output array of shape (n_images, image_size, image_size).
    """
    return fetch_olivetti_faces(shuffle=True).images # random_state=42


def load_mnist_data(split: str, n_per_class: int = None) -> np.ndarray:
    """ 
    Load MNIST 28x28 images from HuggingFace datasets.
    Returns images as vectors with values in [0,1].
    Output array of shape (n_images, image_size, image_size).
    """

    if split not in ["train", "test"]:
        raise ValueError(f"Unknown split '{split}': it must be either 'train' or 'test'.")
    
    dataset = load_dataset("mnist")[split]
    image_size = np.array(dataset[0]["image"]).shape[0]

    labels = np.array([x["label"] for x in dataset])
    total_labels_count = np.bincount(labels)
    min_count = np.min(total_labels_count)
    if n_per_class is None:
        n_per_class = min_count
    else:
        if n_per_class > min_count:
            raise ValueError(f"n_per_class cannot be higher than {min_count}.")
    
    class_to_index = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_index[label].append(idx)
    n_classes = len(class_to_index.keys())
    
    selected_indexes = {
        label: np.random.choice(class_to_index[label], n_per_class, replace=False)
        for label in class_to_index.keys()
    }

    data = np.zeros((n_classes*n_per_class, image_size, image_size))
    for label_idx, (label, indexes) in enumerate(selected_indexes.items()):
        for c, idx in enumerate(indexes):
            data[label_idx*n_per_class + c, :, :] = np.array(dataset[int(idx)]["image"], dtype=np.float32)
    
    data = data / 255.0

    return data


class PatchDataGenerator:

    DATALOADERS = {
        "olivetti": load_olivetti_data,
        "mnist": lambda: load_mnist_data(split="train", n_per_class=1000),
    }

    @classmethod
    def get_dataloader(cls, dataset_name: str):
        """ Return the dataloader function corresponding to the dataset name. """
        if dataset_name not in cls.DATALOADERS.keys():
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        return cls.DATALOADERS[dataset_name]
    
    def __init__(self, dataset_name: str, save_dir: str = None) -> None:
        """
        ARGS:
        - dataset_name: name of the dataset to use. Supported datasets are defined in DATALOADERS.
        - save_dir: directory where to save the plots. If None, plots are not saved.
        """
        self.dataset_name = dataset_name
        self.data_loader = self.get_dataloader(dataset_name)
        self.save_dir = save_dir if save_dir is not None else "outputs/"
        self.patch_size = None
        self.n_patches = None
        self.data = None
  
    def create_patch_dataset(self, patch_size: int = None, n_patches: int = None, return_data: bool = False) -> None | np.ndarray:
        """ 
        ARGS:
            - patch_size: size of the patches to extract from the images (full image if None).
            - n_patches: number of patches to extract from the images (all possible patches if None).
            - return_data: if True, return the patch dataset.
        """

        start_time = time()

        if self.data is not None:
            raise ValueError("A patch dataset was already created and is stored in self.data.")

        dataset = self.data_loader()
        self.patch_size = min(dataset.shape[1], dataset.shape[2]) if patch_size is None else patch_size

        # build all patches
        patches_list = []
        for idx in range(dataset.shape[0]):
            for patch in extract_all_patches(dataset[idx], self.patch_size):
                patches_list.append(patch)

        # randomly select patches
        self.n_patches = len(patches_list) if n_patches is None else n_patches
        self.data = np.zeros((self.patch_size**2, self.n_patches))
        if n_patches is None:
            for patch_idx in range(self.n_patches):
                self.data[:,patch_idx] = patches_list[patch_idx]
        else:
            chosen_patches_idx = np.random.choice(len(patches_list), self.n_patches, replace=False)
            for patch_idx in range(self.n_patches):
                self.data[:,patch_idx] = patches_list[chosen_patches_idx[patch_idx]]

        end_time = time()

        print(f"Patch dataset {self.dataset_name.upper()} created with n_patches={self.n_patches} and patch_size={self.patch_size} [{end_time-start_time:.2f}s]")

        if return_data:
            return self.data
    
    def plot_random_patches(self, n: int = 16, ncol_plot: int = 4, save: bool = False):
        """
        ARGS:
            - n: number of random patches to plot.
            - ncol_plot: number of columns in the plot.
            - save: if True, save the plot in self.save_dir.
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
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f"{self.dataset_name}_random_patches_n={n}.png"), dpi=300)
        else:
            plt.show()
    
    def plot_collection(self, n: int = 500, nrow_plot: int = 10, sort_variance: bool = True, save: bool = False):
        """
        ARGS:
            - n: number of patches to plot.
            - nrow_plot: number of rows in the plot.
            - sort_variance: if True, sort the patches by variance in the plot.
            - save: if True, save the plot in self.save_dir.
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
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f"{self.dataset_name}_collection_n={n}.png"), dpi=300)
        else:
            plt.show()


class PatchDictionary:

    def __init__(self, dict: np.ndarray, dict_name: str = None, save_dir: str = None) -> None:
        """
        ARGS:
            - dict: np.ndarray containing the dictionary values. Each column is an atom.
            - dict_name: name of the dictionary.
            - save_dir: directory where to save the plots.
        """
        self.dict = dict
        self.n_atoms = dict.shape[1]
        self.patch_size = int(np.sqrt(dict.shape[0]))
        if self.patch_size ** 2 != dict.shape[0]:
            raise ValueError("The dictionary doesn't contain square patches in its columns.")
        self.dict_name = dict_name if dict_name is not None else "unknown dict"
        self.save_dir = save_dir if save_dir is not None else "outputs/"
    
    def get_plot_dict(self) -> None:
        """
        DESCRIPTION:
           Unormalize each atom values by a forcing their biggest absolute value to be 1.
           Only useful for visualization purpose because atoms need to be normalized for OMP and KSVD.
        ARGS:
            - except_first_atom: if True, don't normalize the first atom (e.g. DC atom).
        """

        max_abs_values = np.max(np.abs(self.dict), axis=0)
        plot_dict = self.dict / max_abs_values
        plot_dict = (plot_dict + 1) / 2
        return plot_dict
        
    def __plot_dictionary_without_borders(self, ncol_plot: int, transpose_dict: bool, save: bool):

        if ncol_plot is None:
            ncol_plot = int(np.sqrt(self.dict.shape[1]))
            if ncol_plot ** 2 < self.n_atoms: # not a perfect square number of atoms
                ncol_plot += 1
        
        nrow = (self.n_atoms // ncol_plot) + 1 if (self.n_atoms % ncol_plot) > 0 else (self.n_atoms // ncol_plot)

        # saturate atoms and shift values for visualization purpose
        plot_dict = self.get_plot_dict()

        # build the array containing all patches
        collection = np.ndarray((nrow*self.patch_size, ncol_plot*self.patch_size))
        for idx in range(self.n_atoms):
            row = (idx // ncol_plot) * self.patch_size
            col = (idx % ncol_plot) * self.patch_size
            # set patch values in [0,1] for visualization purpose
            patch = plot_dict[:,idx].reshape((self.patch_size, self.patch_size))
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
        
        # saturate atoms and shift values for visualization purpose
        plot_dict = self.get_plot_dict()
        
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
            # set patch values in [0,1] for visualization purpose
            patch = plot_dict[:, patch_idx].reshape((self.patch_size, self.patch_size))
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
        """
        ARGS:
            - ncol_plot: number of columns in the plot.
            - borders: if True, plot the dictionary with borders between patches.
            - transpose_dict: if True, transpose the dictionary representation before plotting it.
            - save: if True, save the plot in self.save_dir.
        """
        if not borders:
            self.__plot_dictionary_without_borders(ncol_plot=ncol_plot, transpose_dict=transpose_dict, save=save)
        else:
            self.__plot_dictionary_with_borders(transpose_dict=transpose_dict, save=save)

    def save_dict(self, filename: str = None, normalize_atoms: bool = True) -> None:

        if filename is None:
            filename = self.dict_name.lower().replace(' ','-') + "_dict.npy"

        if normalize_atoms:
            dict_to_save = self.dict / np.linalg.norm(self.dict, axis=0)

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(self.save_dir, filename), dict_to_save)


if __name__ == "__main__":
    
    # OLIVETTI patch dataset (black and white face pictures)
    data_engine = PatchDataGenerator(dataset_name="olivetti")
    data_engine.create_patch_dataset(patch_size=8, n_patches=1000, return_data=False)
    data_engine.plot_random_patches(save=True)
    data_engine.plot_collection(n=500, nrow_plot=10, sort_variance=True, save=True)

    # MNIST patch dataset (black and white handwritten digits)
    data_engine = PatchDataGenerator(dataset_name="mnist")
    data_engine.create_patch_dataset(patch_size=None, n_patches=None, return_data=False)
    data_engine.plot_random_patches(save=True)
    data_engine.plot_collection(n=400, nrow_plot=20, sort_variance=True, save=True)

    # Haar dictionary (don't normalize atoms for visualization purpose)
    haar_dict = PatchDictionary(
        dict=create_haar_dict(patch_size=8, K=441, normalize_atoms=False, transpose_dict=False),
        dict_name="Haar",
    )
    haar_dict.plot_dictionary(borders=True, save=True)
    haar_dict.save_dict(normalize_atoms=True)

    # DCT dictionary (don't normalize atoms for visualization purpose)
    dct_dict = PatchDictionary(
        dict=create_dct_dict(patch_size=8, K=441, normalize_atoms=False, transpose_dict=True),
        dict_name="DCT",
    )
    dct_dict.plot_dictionary(borders=True, save=True)
    dct_dict.save_dict(normalize_atoms=True)
