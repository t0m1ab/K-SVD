import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_results_synthetic_exp(dir: str = None, n_runs: int = 50, success_threshold: float = 0.01, plot_groups: bool = False, group_size: int = 10):
    """ Plot the results of the synthetic experiment over different noise levels. """

    noise_levels = [0, 10, 20, 30]
    fig, ax = plt.subplots(figsize=(10, 5))

    for noise_db in noise_levels:
        file_path = os.path.join(dir, f"success_scores_{noise_db}dB.npy")
        if not os.path.isfile(file_path):
            print(f"File '{file_path}' was not found...")
            continue
        scores = np.load(file_path)
        n_scores = scores.shape[0]

        if plot_groups and n_scores % group_size == 0:
            sorted_scores = np.sort(scores)
            group_mean_scores = np.mean(sorted_scores.reshape((-1, group_size)), axis=1)
            absc = noise_db * np.ones(group_mean_scores.shape[0])
            ax.scatter(absc, group_mean_scores, marker="s", color="red")
        else:
            absc = noise_db * np.ones(scores.shape[0])
            ax.scatter(absc, scores, marker="o", color="black")
    
    ax.set_title(f"Success scores over {n_runs} runs for the synthetic experiment", size=16)
    ax.set_xticks(noise_levels, ["No noise", "10dB", "20dB", "30dB"])
    ax.set_yticks([25, 30, 35, 40, 45, 50])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(f"success score (threshold={success_threshold})")
    ax.grid(True)
    ax.set_axisbelow(True) # put grid behind the plot
    fig.savefig(os.path.join(dir, "success_scores.png"), dpi=300)
    print(f"Figure 'success_scores.png' saved in '{dir}'!")


def is_power_of_two(n: int) -> bool:
    """
    Check if n is a power of two.
    """
    if (n == 1) or (n == 0):
        return True
    elif n % 2 == 1:
        return False
    else:
        return is_power_of_two(n // 2)


def create_haar_row(n: int) -> np.ndarray:
    """
    Create a pure vertical haar frequencies as patches of size n x n.
    """

    if not is_power_of_two(n):
        raise ValueError("n must be a power of two.")
    
    log_n = int(np.log2(n))
    n_patches = 1 + (n//2) + n*(log_n-1) if log_n > 1 else 2
    haar_row = np.zeros((n, n*n_patches)) # n components for each frequency + continuous component

    # continuous component
    haar_row[:,0:n] = -np.ones((n, n), dtype=np.float32)

    # create patches for each frequency (n patches by frequency except fondamental with n//2 patches)
    half_length = n
    for freq in range(log_n):
        half_length = half_length//2
        positive_positions = np.arange(0, half_length)
        negative_positions = np.arange(half_length, 2*half_length)
        freq_shift = n if freq == 0 else n + n*(n//2) + n*n*(freq-1) # patch_size * number of patches before first patch with frequency freq
        for shift in range(n if freq > 0 else n // 2):
            # absolute positions in haar_row
            pos_col_index = freq_shift + n*shift + positive_positions
            neg_col_index = freq_shift + n*shift + negative_positions
            # define the corresponding patch
            haar_row[:,pos_col_index] = 1.0
            haar_row[:,neg_col_index] = -1.0
            # shift signal one pixel to the right for next patch
            positive_positions = (positive_positions + 1) % n
            negative_positions = (negative_positions + 1) % n
    
    return haar_row

        
def create_haar_dict(patch_size: int, K:int, normalize_atoms: bool = False, transpose_dict: bool = False) -> np.ndarray:
    """
    Create an overcomplete haar dictionary for patches of size patch_size x patch_size.
    Use the trick of the tensor product between pure vertical/horizontal to create the dictionary then converts back to vectors.
    The resulting atoms in the dictionary are not normalized by default.
    """

    n_patches_edge = np.sqrt(K).astype(int)
    if not n_patches_edge**2 == K:
        raise ValueError("K must be a perfect square.")

    # create first row and first column
    vert_freq = create_haar_row(n=patch_size)
    horz_freq = vert_freq.T

    # create the complete haar dictionary with product of vertical and horizontal frequencies
    haar_collection = np.zeros((vert_freq.shape[1], horz_freq.shape[0]))
    haar_collection[:patch_size, :] = vert_freq
    haar_collection[:, :patch_size] = horz_freq
    haar_collection[patch_size:, patch_size:] = horz_freq[patch_size:, :] @ vert_freq[:, patch_size:] / patch_size

    # convert each patch to a n*n vector
    if n_patches_edge != haar_collection.shape[0] // patch_size:
        raise ValueError("Uncomplete implementation for haar dictionary. Please set K=441.")
    n_patches = n_patches_edge ** 2
    haar_dict = np.zeros((patch_size**2, n_patches))
    for row in range(n_patches_edge):
        for col in range(n_patches_edge):
            up = row * patch_size
            left = col * patch_size
            if transpose_dict:
                left, up = up, left
            vector_atom = haar_collection[up:up+patch_size, left:left+patch_size].reshape(-1)
            if normalize_atoms:
                vector_atom = vector_atom / np.linalg.norm(vector_atom)
            haar_dict[:,n_patches_edge*row+col] = vector_atom

    haar_dict[:,0] = np.ones((patch_size**2,)) # set the first atom to the constant vector 1

    # transform values in [-1,1] to [0,1]
    haar_dict = (haar_dict + 1) / 2

    return haar_dict


def create_dct_row(n: int, freq_range: int) -> np.ndarray:
    """
    Create 1D cosine signals for each freq in [0,freq_range[ using DCT-II: https://en.wikipedia.org/wiki/Discrete_cosine_transform
    Order them into a (freq_range, n) matrix and reshape it into a (1, freq_range*n) row vector.
    """
    table = np.array([[np.cos((0.5 + i) * k * np.pi / freq_range) for i in range(n)] for k in range(freq_range)])
    return table.reshape((1,-1))


def create_dct_dict(patch_size: int, K:int, normalize_atoms: bool = False, transpose_dict: bool = False) -> np.ndarray:
    """
    Create an overcomplete dct dictionary containing K patches of size patch_size x patch_size.
    Use the trick of the tensor product between pure vertical/horizontal to create the dictionary then converts back to vectors.
    The resulting atoms in the dictionary are not normalized by default.
    """

    n_patches_edge = np.sqrt(K).astype(int)
    if not n_patches_edge**2 == K:
        raise ValueError("K must be a perfect square.")

    # create dct row and first column
    dct_row = create_dct_row(n=patch_size, freq_range=n_patches_edge)

    # create the overcomplete dct dictionary with product of vertical and horizontal frequencies
    dct_collection = dct_row.T @ dct_row

    # convert each patch to a n*n vector
    dct_dict = np.zeros((patch_size**2, K))
    for row in range(n_patches_edge):
        for col in range(n_patches_edge):
            up = row * patch_size
            left = col * patch_size
            if transpose_dict:
                left, up = up, left
            vector_atom = dct_collection[up:up+patch_size, left:left+patch_size].reshape(-1)
            if normalize_atoms:
                vector_atom = vector_atom / np.linalg.norm(vector_atom)
            dct_dict[:,n_patches_edge*row+col] = vector_atom

    # transform values in [-1,1] to [0,1]
    dct_dict = (dct_dict + 1) / 2

    return dct_dict


if __name__ == "__main__":

    patch_size = 8
    K = 441

    haar_dict = create_haar_dict(patch_size=patch_size, K=K, normalize_atoms=False, transpose_dict=False)
    assert haar_dict.shape == (patch_size**2, K)
    print(f"Haar dictionary for patch size {patch_size} was successfully created and has shape {haar_dict.shape} ")

    dct_dict = create_dct_dict(patch_size=patch_size, K=K, normalize_atoms=False, transpose_dict=True)
    assert dct_dict.shape == (patch_size**2, K)
    print(f"DCT dictionary for patch size {patch_size} was successfully created and has shape {dct_dict.shape} ")