# K-SVD: dictionary design for sparse representation

**Authors:** Tom LABIAUSSE & Grégoire GISSOT

* Implement the K-SVD algorithm described in [K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation](https://legacy.sites.fas.harvard.edu/~cs278/papers/ksvd.pdf)
* Reproduce the experiments and analyze the methods.

### Time Series experiments

* `synthetic_data.py`: create the synthetic data
* `synthetic_exp.py`: run the synthetic experiements described in the paper

### Application to Image Processing

* `patch_data.py`: create overcomplete DCT, Haar and K-SVD dictionaries used to reconstruct images
* `patch_exp.py`: learn an overcomplete dictionary with K-SVD
* `missing_values.py`: run missing values experiments using DCT, Haar or K-SVD dictionary
* `examples/`: contains example images to run the missing values experiments