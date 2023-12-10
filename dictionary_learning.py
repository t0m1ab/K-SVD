import os
import numpy as np
from pathlib import Path

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit


class KSVD:
    """ 
    Perform K-SVD algorithm to learn a dictionary from a set of signals Y as well as the sparse representation of Y using the dictionary .
    """
    def __init__(
        self, 
        n_atoms: int, 
        sparsity: int, 
        pursuit_method: Pursuit, 
        use_dc_atom: bool = False, 
        verbose: bool = False,
        dict: np.ndarray = None,
    ) -> None:
        self.K = n_atoms
        self.sparsity = sparsity
        self.pursuit_method = pursuit_method
        self.dict = dict # if dict is not None, then the KSVD algorithm will start from this dictionary
        self.coeffs = None
        self.dc_atom = use_dc_atom # use the first atom of self.dict as the DC atom and never update it
        self.iteration = None
        self.residual_history = None
        self.verbose = verbose
    
    def fit(
        self, 
        y: np.ndarray, 
        max_iter: int = None, 
        tol: float = None, 
        return_reconstruction: bool = False,
        dict_name: str = None,
        path: str = None,
        save_chekpoints: bool = False,
        checkpoint_step: int = 10,
    ):
        
        if (max_iter is None and tol is None) or (max_iter is not None and tol is not None):
            raise ValueError("Either max_iter or tol must be specified as a stopping rule and you can't specify both.")
        
        if save_chekpoints: # create checkpoints directory if it does not exist
            dict_name = "ksvd_dict" if dict_name is None else dict_name
            path = "outputs/" if path is None else path
            checkpoint_dir = os.path.join(path, dict_name)
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        n, N = y.shape # n_features, n_signals

        # init dictionary from scratch if no dict checkpoint is provided
        if self.dict is None:
            self.dict = np.random.randn(n, self.K)
            self.dict /= np.linalg.norm(self.dict, axis=0)
            if self.dc_atom: # first atom is the DC atom and other atoms are zero mean
                self.dict[:,0] = np.ones(n, dtype=float) / np.sqrt(n)
                self.dict[:,1:] -= np.mean(self.dict[:,1:], axis=0)
        elif self.dict.shape != (n, self.K):
            raise ValueError(f"The shape of the dictionary was expected to be ({n}, {self.K}) but is {self.dict.shape} instead.")

        self.residual_history = []
        stopping_criterion = False
        self.iteration = 0
        while not stopping_criterion:

            # sparse coding stage
            pursuit_algo = self.pursuit_method(dict=self.dict, sparsity=self.sparsity, verbose=False)
            self.coeffs = pursuit_algo.fit(y=y, precompute=False, return_coeffs=True)

            # codebook update stage
            for k in range(int(self.dc_atom), self.K):
                
                selected_indexes = [i for i in range(self.K)]
                selected_indexes.pop(k)
                selected_indexes = np.array(selected_indexes)
                E_k = y - self.dict[:,selected_indexes] @ self.coeffs[selected_indexes,:]

                xk_T_nonzero_indexes = np.where(self.coeffs[k])[0]
                
                if len(xk_T_nonzero_indexes) > 0: # atom k should be used in at least one signal to be updated
                    omega_k = np.zeros((N, xk_T_nonzero_indexes.shape[0]))
                    omega_k[xk_T_nonzero_indexes, np.arange(xk_T_nonzero_indexes.shape[0])] = 1

                    U, delta, Vh = np.linalg.svd(E_k @ omega_k, full_matrices=True)

                    self.dict[:,k] = U[:,0]
                    self.coeffs[k,xk_T_nonzero_indexes] = delta[0] * Vh[0,:]
                else:
                    print(f"Atom {k} was not used in any signal therefore it was not updated during this iteration.")

            residual = np.linalg.norm(y - self.dict @ self.coeffs)
            self.residual_history.append(residual)
            stopping_criterion = self.check_stopping_rule(residual, max_iter, tol)

            if self.verbose:
                total_iter = f"/{max_iter}" if max_iter is not None else ""
                print(f"# Iter {self.iteration}{total_iter} | loss = {residual}")
            
            if save_chekpoints and (self.iteration%checkpoint_step == 0):
                checkpoint_name = f"{dict_name}_iter={self.iteration}.npy"
                res_values_name = f"{dict_name}_iter={self.iteration}_res.npy"
                np.save(os.path.join(checkpoint_dir, checkpoint_name), self.dict)
                np.save(os.path.join(checkpoint_dir, res_values_name), np.array(self.residual_history))
                print(f"Checkpoint [iter={self.iteration}] saved in {os.path.join(checkpoint_dir, checkpoint_name)}")

        if return_reconstruction:
            return self.dict @ self.coeffs
    
    def check_stopping_rule(self, residual: float, max_iter: int, tol: float):
        self.iteration += 1
        if max_iter is not None:
            return self.iteration >= max_iter
        elif tol is not None:
            return residual < tol
        raise ValueError("Either max_iter or tol must be specified as a stopping rule.")


if __name__ == "__main__":

    from synthetic_data import SyntheticData

    # create synthetic data
    data = SyntheticData(n_features=20)
    data.create_synthetic_dictionary(n_atoms=50, normalize_columns=True, return_dict=False)
    y = data.create_synthetic_signals(n_signals=500, sparsity=3, noise_std=0.1, return_signals=True)

    # init KSVD
    # ksvd = KSVD(n_atoms=50, sparsity=3, pursuit_method=SklearnOrthogonalMatchingPursuit, verbose=True)
    ksvd = KSVD(n_atoms=50, sparsity=3, pursuit_method=OrthogonalMatchingPursuit, verbose=True)
    print("KSVD method was successfully initialized!")

    # fit KSVD
    y_pred = ksvd.fit(y=y, max_iter=10)
    print("KSVD method was successfully fitted!")

    # compute score
    score = data.sucess_score(designed_dict=ksvd.dict, tol=0.01)
    print(f"Success score = {score}/{ksvd.K}")