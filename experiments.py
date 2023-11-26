""" 
This file runs the experiments performed in the following paper:
K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation
Michal Aharon - Michael Elad - Alfred Bruckstein
IEEE Transactions on Signal Processing, 2006
"""

from pursuits import SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from dictionary_learning import KSVD
from utils import SyntheticData


def main():

    # create synthetic data
    data = SyntheticData(n_features=20)
    data.create_synthetic_dictionary(n_atoms=50, normalize_columns=True, return_dict=False)
    y = data.create_synthetic_signals(n_signals=1500, sparsity=3, noise_std=0.1, return_signals=True)

    # run KSVD
    ksvd = KSVD(n_atoms=50, sparsity=3, pursuit_method=SklearnOrthogonalMatchingPursuit, verbose=True)
    # ksvd = KSVD(n_atoms=50, sparsity=3, pursuit_method=OrthogonalMatchingPursuit, verbose=True)
    ksvd.fit(y=y, max_iter=80, return_reconstruction=False)

    # compute score
    score = data.sucess_score(designed_dict=ksvd.dict, tol=0.01)

    print(f"Success score = {score}/{ksvd.K}")


if __name__ == "__main__":
    main()