import numpy as np


class SyntheticData:
    
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.dict = None
        self.coeffs = None
        self.signals = None
    
    def create_synthetic_dictionary(self, n_atoms: int, normalize_columns: bool = False, return_dict: bool = False):
        """ 
        Create a synthetic dictionary of $n_atoms$ atoms of dimension $n_features$.
        Each atom is a random vector of dimension $n_features$.
        """
        self.dict = np.random.randn(self.n_features, n_atoms)
        if normalize_columns:
            self.dict /= np.linalg.norm(self.dict, axis=0)
        if return_dict:
            return self.dict

    def create_synthetic_signals(self, n_signals: int, sparsity: int, noise_std: float = 0, return_signals: bool = False):
        """ 
        Create a synthetic set of $n_signals$ signals of dimension $n_features$.
        Each signal is a sparse linear combination of $sparsity$ atoms from $dict$.
        """

        if self.dict is None:
            raise ValueError("You must create a dictionary before creating signals.")

        n_atoms = self.dict.shape[1]
        self.coeffs = np.zeros((n_atoms, n_signals))
        for idx in range(n_signals):
            self.coeffs[np.random.choice(n_atoms, size=sparsity, replace=False), idx] = 1
        self.signals = self.dict @ self.coeffs

        if noise_std > 0:
            noise_sigma = noise_std * np.ones((self.n_features, 1)) @ np.random.rand(1, n_signals)
            self.signals += noise_sigma * np.random.randn(self.n_features, n_signals)

        if return_signals:
            return self.signals
    
    def sucess_score(self, designed_dict: np.ndarray, threshold: float = 0.01):
        """ 
        Compute the success score (see the paper for details) of $designed_dictionary$ with respect to the dictionary $self.dict$.
        """

        if self.dict is None:
            raise ValueError("You must create a dictionary before computing the success score.")

        if self.dict.shape != designed_dict.shape:
            raise ValueError("The shape of the designed dictionary must be the same as the shape of the dictionary created by the class.")

        n_atoms = self.dict.shape[1]
        atoms_matching = -1 * np.ones(n_atoms)

        for col_idx in range(n_atoms):
            distance = np.zeros(n_atoms)
            for i in range(n_atoms):
                distance[i] = 1 - np.abs(self.dict[:,col_idx].T @ designed_dict[:,i])
            idx_matching = np.argmin(distance)
            if atoms_matching[idx_matching] < threshold:
                atoms_matching[idx_matching] = idx_matching
        
        success_score = np.sum(atoms_matching >= 0)
        
        return success_score


if __name__ == "__main__":
        
    data = SyntheticData(n_features=6)
    data.create_synthetic_dictionary(n_atoms=10, normalize_columns=True, return_dict=False)
    y = data.create_synthetic_signals(n_signals=100, sparsity=3, noise_std=0.1, return_signals=True)
    print(f"Synthetic data with shape {y.shape} was successfully created!")

    score = data.sucess_score(designed_dict=data.dict)
    assert score == data.dict.shape[1], "The success score should be equal to the number of atoms in the dictionary when testing the original dictionary."
