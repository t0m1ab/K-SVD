"""
Reconstruction of missing values in an image using OMP with a given dictionary.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from patch_data import PatchDictionary
from utils import (
    convert_255_to_unit_range,
    convert_unit_range_to_255,
    create_dct_dict,
    create_haar_dict,
)


def add_missing_values(signals: np.ndarray, r: float) -> np.ndarray:
    """
    Delete (set to zero) a fraction $r$ of the entries in each column signal in $signals$.
    Returns a mask for each signal (1=no deletion, 0=deletion).
    """
    masks = np.ones(signals.shape, dtype=int)
    p = int(r * signals.shape[0])
    for i in range(signals.shape[1]):
        missing_values_signal = np.random.choice(np.arange(signals.shape[0]), size=p, replace=False)
        masks[missing_values_signal, i] = 0
    return masks


def reconstruct_missing_values(
        signals: np.ndarray,
        masks: np.ndarray,
        pursuit_method: Pursuit,
        dictionary: np.ndarray,
        sparsity: int,
        verbose: bool = False,
    ) -> np.ndarray:
    """
    Reconstruct the missing values in each signal in $signals$ using the mask $mask$ and the dictionary $dictionary$.
    """

    K = dictionary.shape[1] # number of atoms in the dictionary
    n, N = signals.shape
    reconstructed_signals = np.zeros(signals.shape)

    # iter over each signal independently because dictionnary must be normalized 
    # for each signal differently (dependending a given mask)
    for signal_idx in tqdm(range(N)) if verbose else range(N):

        # build signal with missing values
        y = signals[:,signal_idx]
        y_masked = y * masks[:,signal_idx]

        # define masked and non-masked indexes
        masked_indexes = np.where(masks[:,signal_idx] == 0)[0]
        non_masked_indexes = np.where(masks[:,signal_idx] == 1)[0]

        # adapt the dictionary for the signal
        subatom_norms = np.linalg.norm(dictionary[non_masked_indexes, :], axis=0)
        # normalize non-masked subatom and set to zer masked subatom elements
        modified_dict = dictionary.copy() / subatom_norms # normalize non-masked subatom
        modified_dict[masked_indexes, :] = 0.0 # set to zero masked elements ine each atom
        
        # run pursuit algorithm
        pursuit = pursuit_method(dict=modified_dict, sparsity=sparsity, verbose=False)
        signal_coeffs = pursuit.fit(y=y_masked.reshape((-1,1)), return_coeffs=True)
        
        # each atom was renormalized with respect to non-masked elements in the signal
        # in order to reconstruct with the original dictionary, the sparse coefficients need to integrate this normalization        
        signal_coeffs = signal_coeffs / subatom_norms.reshape((-1,1))

        # reconstruct signal thus inferring missing values
        reconstruction = dictionary @ signal_coeffs
        reconstructed_signals[:, signal_idx] = reconstruction.reshape(-1)

    # compute metrics
    reconstruction_errors = signals - reconstructed_signals
    rmse = np.sqrt(np.mean(reconstruction_errors**2))
    mae = np.mean(np.abs(reconstruction_errors))
    metrics = {"rmse": rmse, "mae": mae}
    if verbose:
        print(f"RMSE ({N} samples) = {rmse}")
        print(f"MAE ({N} samples) = {mae}")

    return reconstructed_signals, metrics


def image_to_patches(image: np.ndarray, patch_size: int, crop: bool = False) -> np.ndarray:
    """
    Convert an image to a collection of patches of size patch_size x patch_size.
    If crop is True, the image is cropped to have a width and height divisible by patch_size.
    """

    height, width = image.shape
    n_patches_row = height // patch_size
    n_patches_col = width // patch_size

    if crop:
        image = image[:n_patches_row*patch_size, :n_patches_col*patch_size]
    elif width % patch_size != 0 or height % patch_size != 0:
        raise ValueError(f"Image dimensions ({width},{height}) must be divisible by patch_size ({patch_size},{patch_size}).")

    n_patches = n_patches_row * n_patches_col
    patches = np.zeros((patch_size**2, n_patches)) # each patch is a column vector of size patch_size**2
    for row in range(n_patches_row):
        for col in range(n_patches_col):
            up = row * patch_size
            left = col * patch_size
            patches[:, row*n_patches_col + col] = image[up:up+patch_size, left:left+patch_size].reshape(-1)
    
    return patches


def load_image(image_name: str, image_path: str, resize: tuple = None, save_dir: str = None) -> np.ndarray:
    """
    Load an image from the given path and filename.
    If resize is not None, resize the image to the given dimensions (widht, height).
    If save_version is True, save the resized image in the same directory with the suffix "_{height}x{width}".
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    image = Image.open(image_path) #.convert("L") # grayscale conversion

    if image.mode in ["RGB", "RGBA"]:
        image = image.convert("L") # grayscale conversion (L = Luminance in [0,255])
    elif image.mode != "L":
        raise ValueError(f"Image mode must be RGB, RGBA or L but is {image.mode}.")

    if resize is not None:
        image = image.resize(resize)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        resized_file_name = f"{image_name}_{resize[1]}x{resize[0]}.png"
        image.save(os.path.join(save_dir, resized_file_name))

    np_image = convert_255_to_unit_range(image)

    return np_image


def patches_to_image(patches: np.ndarray, patches_dim: tuple, return_image: bool = False, filename: str = None, path: str = "./images") -> None | np.ndarray:
    """
    Convert a collection of patches into an image. $patches_dim$ indicates how to reorder patches into an image.
    If return_image is True, return the image as a numpy array.
    If filename is not None, then the resulting image in saved at the given path.
    """

    n, N = patches.shape
    patch_size = int(np.sqrt(n))

    if not patch_size**2 == n:
        raise ValueError(f"The dimension of the pacthes must be a perfect square to represent a square patch but n={n} is not.")

    n_patches_row, n_patches_col = patches_dim # number of rows of patches and number of columns of patches
    if not n_patches_row * n_patches_col == N:
        raise ValueError(f"The number of patches ({N}) must be equal to the product of the number of rows ({n_patches_row}) and the number of columns ({n_patches_col}) of patches.")

    np_image = np.zeros((patch_size*n_patches_row, patch_size*n_patches_col), dtype=np.float32)
    for row in range(n_patches_row):
        for col in range(n_patches_col):
            up = row * patch_size
            left = col * patch_size
            patch = patches[:, row*n_patches_col + col].reshape((patch_size, patch_size))
            np_image[up:up+patch_size, left:left+patch_size] = patch

    image = convert_unit_range_to_255(np_image)
    
    if filename is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(os.path.join(path, filename))

    if return_image:
        return image


def fill_missing_values(
        filename: str,
        patch_size: int,
        patch_dimensions: tuple,
        n_atoms: int,
        dictionary_name: str,
        missing_ratio: float,
        pursuit_method: Pursuit = OrthogonalMatchingPursuit,
        sparsity: int = 10,
        save_masked_image: bool = False,
        save_reconstructed_image: bool = False,
        path: str = "./images",
        return_metrics: bool = True,
    ):

    # load image
    expected_size = (patch_dimensions[1] * patch_size, patch_dimensions[0] * patch_size) # (width, height)
    image = load_image(filename=filename, resize=expected_size, save_version=False, path=path) # image with values in [0,1]

    # cut images into non-overlapping patches
    signals = image_to_patches(image=image, patch_size=8)

    # load dictionary
    dictionary = None
    if dictionary_name == "dct":
        dictionary = create_dct_dict(patch_size=patch_size, K=n_atoms, normalize_atoms=False, transpose_dict=True)
    elif dictionary_name == "haar":
        dictionary = create_haar_dict(patch_size=patch_size, K=n_atoms, normalize_atoms=False, transpose_dict=True)
        # add small noise to avoid non invertible matrix when projecting in pursuit algorithm
        dictionary += 1e-6 * np.random.randn(*dictionary.shape)
    else:
        raise ValueError(f"Unknown dictionary name: {dictionary_name}.")
    
    # create missing values mask for each patch
    masks = add_missing_values(signals=signals, r=missing_ratio)

    if save_masked_image: # save masked image
        patches_to_image(
            patches=signals*masks,
            patches_dim=patch_dimensions, 
            return_image=False,
            filename=filename.replace(".png", f"_masked_r={missing_ratio}.png"),
            path=path,
        )

    # reconstruct missing values
    reconstructed_signals, metrics = reconstruct_missing_values(
        signals=signals,
        masks=masks,
        pursuit_method=pursuit_method,
        dictionary=dictionary,
        sparsity=sparsity,
        verbose=False,
    )

    if save_reconstructed_image: # save reconstructed image
        patches_to_image(
            patches=reconstructed_signals,
            patches_dim=patch_dimensions, 
            filename=filename.replace(".png", f"_reconstructed_r={missing_ratio}.png"),
            path=path,
            return_image=False,
        )
    
    if return_metrics:
        return metrics
    

class ImageProcessor():

    def __init__(
            self, 
            patch_size: int, 
            n_atoms: int, 
            pursuit_method: Pursuit, 
            sparsity: int, 
            custom_dicts: dict = None,
            save_dir: str = None,
            verbose: bool = False,
        ) -> None:
        """
        ARGS:
        - patch_size: size of the patches (e.g. 8 for 8x8 patches)
        - n_atoms: number of atoms in the dictionary
        - pursuit_method: pursuit algorithm to use for reconstruction
        - sparsity: max number of atoms used to reconstruct each signal
        - custom_dicts: define custom dictionaries to use for reconstruction (e.g. {"ksvd": "./outputs/ksvd.npy"})
        - save_dir: directory where to save the result images and metrics
        """

        # parameters for image processing
        self.patch_size = patch_size
        self.n_atoms = n_atoms
        self.pursuit_method = pursuit_method
        self.sparsity = sparsity
        self.save_dir = "outputs/" if save_dir is None else save_dir
        self.verbose = verbose
        self.metrics = {}
        
        # create/load dictionaries
        self.dictionaries = {}
        self.dictionaries["dct"] = PatchDictionary(
            dict=create_dct_dict(patch_size=self.patch_size, K=self.n_atoms, normalize_atoms=True),
            dict_name="dct",
        )
        self.dictionaries["haar"] = PatchDictionary(
            dict=create_haar_dict(patch_size=self.patch_size, K=self.n_atoms, normalize_atoms=True),
            dict_name="haar",
        )
        if custom_dicts is not None:
            for dname, dpath in custom_dicts.items():
                if os.path.isfile(dpath):
                    self.dictionaries[dname] = PatchDictionary(
                        dict=np.load(dpath),
                        dict_name=dname,
                    )
                else:
                    raise FileNotFoundError(f"The following dictionary doesn't exist: {dpath}")

        # init temporary image data
        self.reset_image_data()
    
    def reset_image_data(self) -> None:
        """ Reset temporary image data. """
        self.image_name = None
        self.image = None
        self.patch_dim = None
        self.signals = None
        self.missing_ratio = None
        self.masks = None
        self.reconstructed_image = None

    def load_image(self, image_name: str, image_path: str, patch_dim: tuple, return_patches: bool = False) -> None | np.ndarray:
        """
        ARGS:
        - image_name: name/tag of the image (e.g. "lena")
        - image_path: path to the image file (e.g. "./examples/lena.png")
        - patch_dim: dimension of the image in patches (e.g. (27, 22) for 27 rows and 22 columns of patches in the image)
        - return_patches: if True, return the patch representation of the image as a numpy array
        """
        
        # load image (resize if necessary) with values in [0,1] and save the resized version in outputs/image_name/
        self.image = load_image(
            image_name=image_name,
            image_path=image_path,
            resize=(patch_dim[1] * self.patch_size, patch_dim[0] * self.patch_size), # (width, height)
            save_dir=os.path.join(self.save_dir, image_name),
        )
        self.image_name = image_name
        self.patch_dim = patch_dim

        # split images into non-overlapping patches and store them as column vectors in self.signals
        self.signals = image_to_patches(image=self.image, patch_size=self.patch_size)

        if return_patches:
            return self.signals
    
    def mask_image(self, missing_ratio: float, save_masked_image: bool = False, return_masks: bool = False) -> None | np.ndarray:
        """
        ARGS:
        - missing_ratio: fraction of missing values in the image (e.g. 0.5 for 50% of missing values)
        - save_masked_image: if True, save the masked image in directory {self.save_dir}}/{self.image_name}/
        - return_masks: if True, return the mask for each patch as a numpy array
        """
        
        if self.image is None:
            raise ValueError("No image loaded. Please load an image with load_image() before masking it.")
        
        # create missing values mask for each patch
        self.missing_ratio = missing_ratio
        self.masks = add_missing_values(signals=self.signals, r=self.missing_ratio)

        if save_masked_image:
            patches_to_image(
                patches=self.signals*self.masks,
                patches_dim=self.patch_dim,
                filename=f"{self.image_name}_masked_r={self.missing_ratio}.png",
                path=os.path.join(self.save_dir, self.image_name),
                return_image=False,
            )
        
        if return_masks:
            return self.masks
    
    def reconstruct_image(self, dict_name: str, save_metrics: bool = False, save_rec_image: bool = False) -> None:
        """
        ARGS:
        - dict_name: name of the dictionary to use for reconstruction (e.g. "dct" or "haar")
        - save_metrics: if True, save the metrics computed on the image reconstructions
        - save_rec_image: if True, save the reconstructed image in directory {self.save_dir}}/{self.image_name}/
        """

        if self.signals is None or self.masks is None:
            raise ValueError("No image or masks loaded.")

        if dict_name not in self.dictionaries.keys():
            raise ValueError(f"Unknown dictionary name: {dict_name}. Available dictionaries are {self.dictionaries.keys()}.")

        # reconstruct missing values
        if self.verbose:
            print(f"Reconstructing image {self.image_name.upper()} with dictionary {dict_name.upper()} and missing ratio {self.missing_ratio} ...")
        rec_signals, rec_metrics = reconstruct_missing_values(
            signals=self.signals,
            masks=self.masks,
            pursuit_method=self.pursuit_method,
            dictionary=self.dictionaries[dict_name].dict,
            sparsity=self.sparsity,
            verbose=self.verbose,
        )

        if save_metrics:
            if dict_name not in self.metrics.keys():
                self.metrics[dict_name] = {}
            self.metrics[dict_name][self.missing_ratio] = rec_metrics

        if save_rec_image: # save reconstructed image
            patches_to_image(
                patches=rec_signals,
                patches_dim=self.patch_dim, 
                filename=f"{self.image_name}_{dict_name.upper()}_r={self.missing_ratio}.png",
                path=os.path.join(self.save_dir, self.image_name),
                return_image=False,
            )
    
    def plot_metrics(self, dict_name: str = None) -> None:
        """
        ARGS:
        - dict_name: name of the dictionary to plot the metrics for (e.g. "dct" or "haar"). If None, plot the metrics for all dictionaries.
        """

        if dict_name is not None and not dict_name in self.metrics.keys():
            raise ValueError(f"Unknown dictionary name: {dict_name}. Metrics were computed only for dictionaries: {self.metrics.keys()}.")

        dnames = [dict_name] if dict_name is not None else self.metrics.keys()

        # plot RMSE for each dictionary
        for dname in dnames:
            mratios = sorted(list(self.metrics[dname].keys()))
            plt.plot(
                self.metrics[dname].keys(),
                [self.metrics[dname][r]["rmse"] for r in mratios],
                linestyle='--',
                marker='o',
                label=f"RMSE {dname}",
            )

        plt.xlabel("Ratio of missing pixels in the image")
        plt.ylabel("RMSE")
        plt.title(f"Sparse reconstruction of {self.image_name.upper()}", fontsize=16)
        plt.legend(loc="upper left")

        plotname = f"rmse_{self.image_name}.png" if dict_name is None else f"rmse_{dict_name}_{self.image_name}.png"
        plt.savefig(fname=os.path.join(self.save_dir, self.image_name, plotname), dpi=300)
        

def main(image_name: str = "lena"):

    # LOAD KSVD DICTIONARIES
    processor = ImageProcessor(
        patch_size=8,
        n_atoms=441,
        pursuit_method=OrthogonalMatchingPursuit,
        sparsity=10,
        custom_dicts={
            "ksvd_olivetti": "./outputs/ksvd_olivetti/ksvd_olivetti.npy",
        },
        verbose=True,
    )

    # LOAD IMAGE
    processor.load_image(
        image_name=image_name,
        image_path=f"./examples/{image_name}.png",
        patch_dim=(27, 22),
    )

    # RECONSTRUCT IMAGE USING DIFFERENT DICTIONARIES
    for dname in processor.dictionaries.keys(): # dct - haar - ksvd_olivetti

        for mratio in [0.2, 0.4, 0.5, 0.6, 0.7]:

            processor.mask_image(
                missing_ratio=mratio,
                save_masked_image=True,
            )

            processor.reconstruct_image(
                dict_name=dname,
                save_metrics=True,
                save_rec_image=True,
            )
    
    # PLOT METRICS FOR ALL RECONSTRUCTIONS
    processor.plot_metrics()


if __name__ == "__main__":
    main(image_name="lena") # AVAILABLE IMAGES: shannon - keogh - lena