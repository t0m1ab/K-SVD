import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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

        
def create_haar_dict(patch_size: int):
    """
    Create a complete haar dictionary of size n x n.
    Use the trick of the tensor product between pure vertical/horizontal to create the dictionary then converts back to vectors.
    """

    # create first row and first column
    vert_freq = create_haar_row(n=patch_size)
    horz_freq = vert_freq.T

    # create the complete haar dictionary with product of vertical and horizontal frequencies
    haar_collection = np.zeros((vert_freq.shape[1], horz_freq.shape[0]))
    haar_collection[:patch_size, :] = vert_freq
    haar_collection[:, :patch_size] = horz_freq
    haar_collection[patch_size:, patch_size:] = horz_freq[patch_size:, :] @ vert_freq[:, patch_size:] / patch_size

    # convert each patch to a n*n vector
    n_patches_edge = haar_collection.shape[0] // patch_size
    n_patches = n_patches_edge ** 2
    haar_dict = np.zeros((patch_size**2, n_patches))
    for row in range(n_patches_edge):
        for col in range(n_patches_edge):
            up = row * patch_size
            left = col * patch_size
            haar_dict[:,n_patches_edge*row+col] = haar_collection[up:up+patch_size, left:left+patch_size].reshape(-1)

    return haar_dict


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


if __name__ == "__main__":
    
    # plot_results_synthetic_exp(
    #     dir="synthetic_experiments/",
    #     n_runs=50,
    #     success_threshold=0.01
    # )

    haar_dict = create_haar_dict(patch_size=8)
    print(f"Haar dictionnary for patch size 8 was successfully created and has shape {haar_dict.shape} ")