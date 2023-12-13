"""
Data loader for the Articulary Word Recognition dataset.
Available at: https://timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition
Download the dataset and put the a folder named "ArticularyWordRecognition" in the same directory as this file.
"""

import os
from pathlib import Path
import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN

from pursuits import Pursuit, SklearnOrthogonalMatchingPursuit, OrthogonalMatchingPursuit
from dictionary_learning import KSVD


class AWRDataloader():

    def __init__(
            self, 
            dataset_path: str, 
            dataset_name: str = None
        ) -> None:

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.dataset_name = "ArticularyWordRecognition" if dataset_name is  None else dataset_name
        
        self.dimension = None
        self.splits = None
        self.data_filename = None
        self.signals = None
        self.labels = None

    def load(
            self, 
            dimension: int, 
            split: str = None, 
            return_data: bool = False
        ) -> None | tuple[np.ndarray, np.ndarray]:
        """ Load the corresponding set of signals. """

        if dimension not in range(1,10):
            raise ValueError(f"Dimension {dimension} not in range [1,9].")
        self.dimension = dimension

        if split is not None and not split in ["train", "test"]:
            raise ValueError(f"Split {split} not in ['train', 'test'].")
        self.splits = [split] if split is not None else ["train", "test"]

        self.signals = {}
        self.labels = {}
        for s in self.splits:
            # define filename and path
            data_filename = f"ArticularyWordRecognitionDimension{self.dimension}_{s.upper()}.arff"
            data_file_path = os.path.join(self.dataset_path, data_filename)
            # load data
            data = pd.DataFrame(arff.loadarff(data_file_path)[0]).to_numpy(dtype=np.float32)
            assert data.shape[1] == 145
            # store data
            self.signals[s] = data[:, :-1].copy().T
            self.labels[s] = data[:, -1].copy()
            print(f"{s.upper()} signals shape: {self.signals[s].shape}")
            print(f"{s.upper()} labels shape: {self.labels[s].shape}")

        if return_data:
            return self.signals, self.labels
    
    def plot_signals(self, split: str, pred: np.ndarray = None) -> None:
        """ Plot some signals from the dataset. """

        if pred is not None and pred.shape != self.signals[split].shape:
            raise ValueError(f"Predictions shape {pred.shape} does not match signals shape {self.signals[split].shape}.")

        random_index = np.random.choice(self.signals[split].shape[1], size=9, replace=False)
        print(f"Plotting signals {random_index} from {split.upper()} split...")

        fig, ax = plt.subplots(3, 3, figsize=(20, 10))
        for i, idx in enumerate(random_index):
            ax[i//3, i%3].plot(self.signals[split][:,idx], label="original")
            if pred is not None:
                ax[i//3, i%3].plot(pred[:,idx], label="reconstruction")
            ax[i//3, i%3].set_title(self.labels[split][idx])
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    awr = AWRDataloader(
        dataset_path="ArticularyWordRecognition/", 
        dataset_name="AWR"
    )

    signals, labels = awr.load(dimension=5, split=None, return_data=True)

    # init KSVD
    ksvd = KSVD(n_atoms=25, sparsity=5, pursuit_method=OrthogonalMatchingPursuit, verbose=True)
    print("KSVD method was successfully initialized!")

    # fit KSVD
    sparse_signals_train = ksvd.fit(y=signals["train"], max_iter=50, return_reconstruction=True)
    print("KSVD method was successfully fitted!")

    # awr.plot_signals(split="train", pred=sparse_signals_train)

    # use KSVD to build sparse representations of test signals
    print(f"Test set has shape {signals['test'].shape}")
    OMP = OrthogonalMatchingPursuit(sparsity=5, dict=ksvd.dict, verbose=True)
    sparse_signals_test = ksvd.dict @ OMP.fit(y=signals["test"], return_coeffs=True)

    # fit KNN
    neigh = KNN(n_neighbors=3)
    neigh.fit(sparse_signals_train.T, labels["train"].T)

    # predict on test set using KNN
    y_pred_test = neigh.predict(sparse_signals_test.T)

    print(f"Class balance in test set: {np.bincount(labels['test'].astype(int))}")   
    print(f"KNN accuracy = {np.mean(labels['test'] == y_pred_test):.3f}")

    # print(labels["test"])
    # print(y_pred_test)

    # dim=5, n_atoms=25, sparsity=5, max_iter=50, KNN=3 => acc=0.72


