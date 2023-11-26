import numpy as np

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit



class KSVD:
    """ Perform K-SVD algorithm to learn a dictionary from a set of signals Y as well as the sparse representation of Y using the dictionary ."""
    def __init__(self, n_atoms: int, sparsity: int, pursuit_method: Pursuit, verbose: bool = False) -> None:
        self.K = n_atoms
        self.sparsity = sparsity
        self.pursuit_method = pursuit_method
        self.dict = None
        self.coeffs = None
        self.iteration = None
        self.verbose = verbose
    
    def fit(self, y: np.ndarray, max_iter: int = None, tol: float = None, return_reconstruction: bool = False):
        
        if (max_iter is None and tol is None) or (max_iter is not None and tol is not None):
            raise ValueError("Either max_iter or tol must be specified as a stopping rule and you can't specify both.")
        
        n, N = y.shape # n_features, n_signals
        self.dict = np.random.randn(n, self.K)
        self.dict /= np.linalg.norm(self.dict, axis=0)

        stopping_criterion = False
        self.iteration = 0
        while not stopping_criterion:

            # sparse coding stage
            pursuit_algo = self.pursuit_method(dict=self.dict, sparsity=self.sparsity, verbose=False)
            self.coeffs = pursuit_algo.fit(y=y, precompute=False, return_coeffs=True)

            # codebook update stage
            for k in range(self.K):
                
                selected_indexes = [i for i in range(self.K)]
                selected_indexes.pop(k)
                selected_indexes = np.array(selected_indexes)
                E_k = y - self.dict[:,selected_indexes] @ self.coeffs[selected_indexes,:]

                xk_T_nonzero_indexes = np.where(self.coeffs[k])[0]
                omega_k = np.zeros((N, xk_T_nonzero_indexes.shape[0]))
                omega_k[xk_T_nonzero_indexes, np.arange(xk_T_nonzero_indexes.shape[0])] = 1

                U, delta, Vh = np.linalg.svd(E_k @ omega_k, full_matrices=True)

                d_k = U[:,0]
                first_singular_value = delta[0] if len(delta.shape) > 0 else 1
                x_k = first_singular_value * Vh[0,:]

                self.dict[:,k] = d_k
                self.coeffs[k,xk_T_nonzero_indexes] = x_k
            
            residual = np.linalg.norm(y - self.dict @ self.coeffs)
            stopping_criterion = self.check_stopping_rule(residual, max_iter, tol)

            if self.verbose:
                total_iter = f"/{max_iter}" if max_iter is not None else ""
                print(f"# Iter {self.iteration}{total_iter} | residual = {residual}")
        
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

    from utils import SyntheticData

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