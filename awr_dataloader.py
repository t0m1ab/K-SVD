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
        self.split = None
        self.data_filename = None
        self.signals = None
        self.labels = None

    def load(
            self, 
            dimension: int, 
            split: str = "train", 
            return_data: bool = False
        ) -> None | tuple[np.ndarray, np.ndarray]:
        """ Load the corresponding set of signals. """

        if dimension not in range(1,10):
            raise ValueError(f"Dimension {dimension} not in range [1,9].")
        self.dimension = dimension

        if not split in ["train", "test"]:
            raise ValueError(f"Split {split} not in ['train', 'test'].")
        self.split = split

        self.data_filename = f"ArticularyWordRecognitionDimension{self.dimension}_{self.split.upper()}.arff"
        data_file_path = os.path.join(self.dataset_path, self.data_filename)

        data = pd.DataFrame(arff.loadarff(data_file_path)[0]).to_numpy(dtype=np.float32)
        assert data.shape == (275, 145)

        self.signals = data[:, :-1].copy().T
        self.labels = data[:, -1].copy()

        print(f"Signals shape: {self.signals.shape}")
        print(f"Labels shape: {self.labels.shape}")

        if return_data:
            return self.signals, self.labels
    
    def plot_signals(self, pred: np.ndarray = None) -> None:
        """ Plot some signals from the dataset. """

        if pred is not None and pred.shape != self.signals.shape:
            raise ValueError(f"Predictions shape {pred.shape} does not match signals shape {self.signals.shape}.")

        fig, ax = plt.subplots(3, 3, figsize=(20, 10))
        for i in range(9):
            ax[i//3, i%3].plot(signals[:,i], label="original")
            if pred is not None:
                ax[i//3, i%3].plot(pred[:,i], label="reconstruction")
            ax[i//3, i%3].set_title(labels[i])
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    awr = AWRDataloader(
        dataset_path="ArticularyWordRecognition/", 
        dataset_name="AWR"
    )

    signals, labels = awr.load(dimension=2, split="train", return_data=True)

    # print(signals.dtype)
    # print(signals.shape)
    # print(labels)

    # init KSVD
    ksvd = KSVD(n_atoms=25, sparsity=10, pursuit_method=OrthogonalMatchingPursuit, verbose=True)
    print("KSVD method was successfully initialized!")

    # fit KSVD
    y_pred = ksvd.fit(y=signals, max_iter=100, return_reconstruction=True)
    print("KSVD method was successfully fitted!")

    awr.plot_signals(pred=y_pred)
