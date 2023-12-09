"""
Code for reconstructing missing values in signals using a given dictionary and OMP algorithm.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from utils import create_dct_dict, create_haar_dict


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
    for signal_idx in tqdm(range(N)):

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
        pursuit = pursuit_method(dict=modified_dict, sparsity=sparsity, verbose=verbose)
        signal_coeffs = pursuit.fit(y=y_masked.reshape((-1,1)), return_coeffs=True, verbose=verbose)
        
        # each atom was renormalized with respect to non-masked elements in the signal
        # in order to reconstruct with the original dictionary, the sparse coefficients need to integrate this normalization        
        signal_coeffs = signal_coeffs / subatom_norms.reshape((-1,1))

        # reconstruct signal thus inferring missing values
        reconstruction = dictionary @ signal_coeffs
        reconstructed_signals[:, signal_idx] = reconstruction.reshape(-1)

    # compute reconstruction error
    signals_rec_errors = np.sqrt( (np.linalg.norm(signals - reconstructed_signals, axis=0)**2) / (n**2) )
    mean_rec_error = np.mean(signals_rec_errors)
    print(f"Mean reconstruction error ({N} samples) = {mean_rec_error}")

    return reconstructed_signals, mean_rec_error


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


def load_image(filename: str, path: str = "./images", resize: tuple = None, save_version: bool = False) -> np.ndarray:
    """
    Load an image from the given path and filename.
    If resize is not None, resize the image to the given dimensions (widht, height).
    If save_version is True, save the resized image in the same directory with the suffix "_{height}x{width}".
    """

    file_path = os.path.join(path, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No file named {filename} in {path}")
    
    image = Image.open(file_path) #.convert("L") # grayscale conversion

    if image.mode in ["RGB", "RGBA"]:
        image = image.convert("L") # grayscale conversion (L = Luminance in [0,255])
    elif image.mode != "L":
        raise ValueError(f"Image mode must be RGB, RGBA or L but is {image.mode}.")

    if resize is not None:
        image = image.resize(resize)

    if save_version:
        version_name = f"{filename.replace('.png','')}_{resize[1]}x{resize[0]}.png"
        image.save(os.path.join(path, version_name))

    np_image = convert_image_255_to_unit_range(image)

    return np_image


def convert_image_255_to_unit_range(image: np.ndarray) -> np.ndarray:
    """
    Convert image with values in range [0,255] to image with values in range [-1,1].
    """
    return np.array(image, dtype=np.float32) / 255


def convert_image_unit_range_to_255(image: np.ndarray) -> np.ndarray:
    """
    Convert image with values in range [-1,1] to image with values in range [0,255].
    """
    return np.array(255 * image, dtype=np.uint8)


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

    image = convert_image_unit_range_to_255(np_image)
    
    if filename is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(image)
        image.save(os.path.join(path, filename))

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
        return_mean_error: bool = True,
    ):

    # load image
    expected_size = (patch_dimensions[1] * patch_size, patch_dimensions[0] * patch_size) # (width, height)
    image = load_image(filename=filename, resize=expected_size, save_version=False, path=path)

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
    reconstructed_signals, reconstruction_error = reconstruct_missing_values(
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
            return_image=False, 
            filename=filename.replace(".png", f"_reconstructed_r={missing_ratio}.png"),
            path=path,
        )
    
    if return_mean_error:
        return reconstruction_error


if __name__ == "__main__":

    # 27 x 22 = 594
    # 27 x 8 = 216
    # 22 x 8 = 176

    filename = "keogh.png" # "keogh.png"
    patch_size = 8
    patch_dimensions = (27, 22)
    n_atoms = 441
    dictionary_name = "haar"

    rec_errors = []
    for missing_ratio in [0.2, 0.4, 0.5, 0.6, 0.7, 0.9]:

        error = fill_missing_values(
            filename=filename,
            patch_size=patch_size,
            patch_dimensions=patch_dimensions,
            n_atoms=n_atoms,
            dictionary_name=dictionary_name,
            missing_ratio=missing_ratio,
            pursuit_method=OrthogonalMatchingPursuit,
            sparsity=10,
            save_masked_image=True,
            save_reconstructed_image=True,
            path=f"./images/{dictionary_name}/",
            return_mean_error=True,
        )

        rec_errors.append(error)
    
    plt.plot(rec_errors)
    plt.savefig(f"./images/{dictionary_name}/rec_errors.png")