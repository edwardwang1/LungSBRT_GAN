from torch.utils.data import Dataset
import os
import numpy as np

class VolumesFromList(Dataset):
    def __init__(self, dataDirectory, list_path):
        self.dataDirectory = dataDirectory

        with open(list_path) as f:
            self.lines = f.read().splitlines()

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.lines)

    def __getitem__(self, idx):
        volumes = np.load(os.path.join(self.dataDirectory, self.lines[idx] + ".npy"))
        return volumes
