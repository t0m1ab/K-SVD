import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from sklearn.linear_model import orthogonal_mp


class Pursuit():
    """ 
    Class of pursuit algorithms solving the following category of problems: 
        $$ \min_{x} ||y - Ax||_2^2 $$ such that $||x||_0 \leq T_0$
    - $A$ is a dictionary with atoms as columns
    - $y$ is a signal to decompose as a column vector
    - $T_0$ is the sparsity constraint (max number of atoms from the dictionary to use for reconstruction)

    $y$ can be a matrix containing multiple signals as its columns and $x$ the corresponding matrix giving the sparse representation.
    """
    def __init__(self, dict: np.ndarray, sparsity: int = None, verbose: bool = False) -> None:
        self.dict = dict
        self.sparsity = sparsity
        self.is_fit = False
        self.verbose = verbose
    
    def fit(self, y: np.ndarray):
        """
        Solve the pursuit problem for the signal $y$, the dictionary $self.dict$ and the sparsity constraint $self.sparsity$.
        """
        raise NotImplementedError("This method should be implemented in a child class.")


class SklearnOrthogonalMatchingPursuit(Pursuit):

    def __init__(self, dict: np.ndarray, sparsity: int = None, verbose: bool = False) -> None:
        super().__init__(dict, sparsity, verbose)
    
    def fit(self, y: np.ndarray, precompute: bool = False, force: bool = False, return_coeffs: bool = False, **kwargs):
        n = self.dict.shape[0]

        if y.shape[0] != n:
            raise ValueError(f"y should have {n} rows (like self.dict which has the shape {self.dict.shape}), but has {y.shape[0]} rows instead.")
        
        if self.is_fit and not force:
            raise ValueError("OMP.fit() has already been run. If you run it again, you will loose the previous results. Set force=True to run it anyway.")

        self.is_fit = False
        self.runtime = time()
        result = orthogonal_mp(X=self.dict, y=y, n_nonzero_coefs=self.sparsity, precompute=precompute)
        self.is_fit = True
        self.runtime = time() - self.runtime

        if return_coeffs:
            return result


class OrthogonalMatchingPursuit(Pursuit):

    def __init__(self, dict: np.ndarray, sparsity: int = None, verbose: bool = False) -> None:
        super().__init__(dict, sparsity, verbose)
        self.signal = None
        self.support = None
        self.coeffs = None
        self.selection_order = None
        self.res_norms = None
        self.reconstructions = None

    def fit(self, y: np.ndarray, force: bool = False, return_coeffs: bool = False, **kwargs):

        n = self.dict.shape[0] # n_features
        K = self.dict.shape[1] # n_atoms
        N = y.shape[1] # n_signals

        if y.shape[0] != n:
            raise ValueError(f"y should have {n} rows (like self.dict which has the shape {self.dict.shape}), but has {y.shape[0]} rows instead.")
        
        if self.is_fit and not force:
            raise ValueError("OMP.fit() has already been run. If you run it again, you will loose the previous results. Set force=True to run it anyway.")

        self.y = y
        residual = y.copy()
        self.coeffs = np.zeros((K, N), dtype=float)
        self.support = np.zeros((K, N), dtype=bool)
        self.selection_order = np.zeros((self.sparsity, N), dtype=int)
        self.res_norms = np.zeros((self.sparsity+1, N), dtype=float)
        self.res_norms[0] = np.linalg.norm(residual, axis=0)
        self.reconstructions = np.zeros((n, y.shape[1], self.sparsity), dtype=float)
        self.is_fit = False
        self.runtime = time()
        
        for iteration in tqdm(range(self.sparsity), desc="OMP iterations") if self.verbose else range(self.sparsity):

            # find best atom
            projections = self.dict.T @ residual # projections[i,j] is the projection of the j-th signal on the i-th atom
            self.selection_order[iteration] = np.argmax(np.abs(projections), axis=0) # argmax on each column
            self.support[self.selection_order[iteration], np.arange(N)] = True

            # coefficients update with orthogonal projection
            for sample_idx in range(N):
                indexes = np.where(self.support[:, sample_idx].reshape(-1))[0] # indexes as a row
                vect = self.dict[:, indexes]
                self.coeffs[indexes, sample_idx] = np.linalg.pinv(vect.T @ vect) @ vect.T @ y[:, sample_idx]

            # residual update
            reconstruction = self.dict @ self.coeffs
            self.reconstructions[:, :, iteration] = reconstruction
            residual = y - reconstruction
            self.res_norms[iteration + 1] = np.linalg.norm(residual, axis=0)
        
        self.is_fit = True
        self.runtime = time() - self.runtime

        if return_coeffs:
            return self.coeffs

    def plot_residual_norms(self, plot: bool = True, save: bool = False, save_dir: str = None):
        """ Plot the evolution of each residual norm during the OMP iterations """

        if not self.is_fit:
            raise ValueError("The OMP algorithm has not been run yet. Please run OMP.fit() first.")
        
        fig, ax = plt.subplots()
        for res_idx in range(self.res_norms.shape[1]):
            res_values = self.res_norms[:, res_idx] / self.res_norms[0, res_idx]
            ax.plot(res_values, marker="o")
        mean_res_values = np.mean(self.res_norms, axis=1) / np.mean(self.res_norms[0])
        ax.plot(mean_res_values, linestyle="--", color="black", label="mean")
        ax.set_xlabel("step")
        ax.set_ylabel("Ratio residual_norm / signal_norm")
        ax.title.set_text("Residuals norm evolution during OMP iterations")

        if plot:
            plt.show()
        if save:
            if save_dir is None:
                save_dir = "default_plots/"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "residual_norms.png"))


if __name__ == "__main__":
    
    dict = np.random.randn(10, 5)
    y = np.random.randn(10, 10)

    # sklearn OMP test
    sklearn_omp = SklearnOrthogonalMatchingPursuit(dict=dict, sparsity=3, verbose=False)
    print("SklearnOMP method was successfully initialized!")
    sklearn_omp.fit(y=y, precompute=False)
    print(f"SklearnOMP fit method didn't crash! (runtime={sklearn_omp.runtime:.3f}s)")

    # custom OMP test
    omp = OrthogonalMatchingPursuit(dict=dict, sparsity=3, verbose=False)
    print("SklearnOMP method was successfully initialized!")
    omp.fit(y=y, precompute=False)
    print(f"SklearnOMP fit method didn't crash! (runtime={omp.runtime:.3f}s)")
    # omp.plot_residual_norms(plot=False, save=True, save_dir="images/")