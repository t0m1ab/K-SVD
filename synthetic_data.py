import numpy as np


class SyntheticDataGenerator:
    
    def __init__(self, n_features: int) -> None:
        """
        ARGS:
            - n_features: dimension of the signals
        """
        self.n_features = n_features
        self.dict = None
        self.coeffs = None
        self.signals = None
    
    def create_synthetic_dictionary(self, n_atoms: int, normalize_columns: bool = False, return_dict: bool = False):
        """ 
        DESCRIPTION:
            Create a synthetic dictionary of $n_atoms$ atoms of dimension $n_features$.
            Each atom is a random vector of dimension $n_features$.
        ARGS:
            - n_atoms: number of atoms in the dictionary
            - normalize_columns: if True, normalize the columns of the dictionary
            - return_dict: if True, return the dictionary
        """

        self.dict = np.random.uniform(low=-1.0, high=1.0, size=(self.n_features, n_atoms))
        if normalize_columns:
            self.dict /= np.linalg.norm(self.dict, axis=0)
        if return_dict:
            return self.dict

    def create_synthetic_signals(self, n_signals: int, sparsity: int, noise_std: float = 0, return_signals: bool = False):
        """ 
        DESCRIPTION:
            Create a synthetic set of $n_signals$ signals of dimension $n_features$.
            Each signal is a sparse linear combination of $sparsity$ atoms from $dict$.
        ARGS:
            - n_signals: number of signals to create
            - sparsity: number of atoms used to construct each signal
            - noise_std: standard deviation of the noise added to the signals
            - return_signals: if True, return the signals
        """

        if self.dict is None:
            raise ValueError("You must create a dictionary before creating signals.")

        n_atoms = self.dict.shape[1]
        self.coeffs = np.zeros((n_atoms, n_signals))
        for idx in range(n_signals):
            self.coeffs[np.random.choice(n_atoms, size=sparsity, replace=False), idx] = 1
        self.signals = self.dict @ self.coeffs

        if noise_std > 0:
            # print(f"Noise std added to the signals = {noise_std}")
            self.signals += np.random.normal(scale=noise_std, size=(self.n_features, n_signals))

        if return_signals:
            return self.signals
    
    def sucess_score(self, designed_dict: np.ndarray, threshold: float = 0.01):
        """ 
        DESCRIPTION:
            Compute the success score (see the paper for details) of $designed_dictionary$ with respect to the dictionary $self.dict$.
        ARGS:
            - designed_dict: the dictionary to test
            - threshold: the threshold to use to determine if two atoms are matching
        """

        if self.dict is None:
            raise ValueError("You must create a dictionary before computing the success score.")

        if self.dict.shape != designed_dict.shape:
            raise ValueError("The shape of the designed dictionary must be the same as the shape of the dictionary created by the class.")

        n_atoms = self.dict.shape[1]
        atoms_matching = -1 * np.ones(n_atoms)

        # iter over each learned atom to find a matching atom in the designed dictionary
        for col_idx in range(n_atoms):
            distance = np.ones(n_atoms) - np.abs(self.dict[:,col_idx].T @ designed_dict) # positive distances thks to normalization
            idx_matching = np.argmin(distance)
            if distance[idx_matching] < threshold:
                atoms_matching[col_idx] = idx_matching
        
        success_score = np.sum(atoms_matching >= 0)
        
        return success_score


if __name__ == "__main__":
    
    # Test the creation of a synthetic dictionary
    data_engine = SyntheticDataGenerator(n_features=6)
    data_engine.create_synthetic_dictionary(n_atoms=10, normalize_columns=True, return_dict=False)
    y = data_engine.create_synthetic_signals(n_signals=100, sparsity=3, noise_std=0.1, return_signals=True)
    print(f"Synthetic data with shape {y.shape} was successfully created!")

    # Test the success score
    score = data_engine.sucess_score(designed_dict=data_engine.dict)
    assert score == data_engine.dict.shape[1], "The success score should be equal to the number of atoms in the dictionary when testing the original dictionary."
