from torch.utils.data import Dataset
import os
import numpy as np

class Volumes(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.dir_files = os.listdir(self.directory)

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.dir_files)

    def __getitem__(self, idx):
        # Load the image from the file
        # Filename based on the index
        volumes = np.load(os.path.join(self.directory, self.dir_files[idx]))

        return volumes


class VolumesFromList(Dataset):
    def __init__(self, dataDirectory, patientListDirectory, singleLesionList, valFold, testingHoldoutFold, test=False, holdout=False):
        self.dataDirectory = dataDirectory
        self.test = test
        self.holdout = holdout
        self.trainIDs = []
        self.valIDs = []
        self.testIDs = []

        with open(singleLesionList) as f:
            lines = f.read().splitlines()
            for l in lines:
                self.trainIDs.append(l)

        for i in range(5):
            filePath = os.path.join(patientListDirectory, "fold" + str(i) + ".txt")
            with open(filePath) as f:
                lines = f.read().splitlines()
            if i == valFold:
                for l in lines:
                    self.valIDs.append(l)
            elif i == testingHoldoutFold:
                for l in lines:
                    self.testIDs.append(l)
            else: #train
                for l in lines:
                    self.trainIDs.append(l)

    def __len__(self):  # The length of the dataset is important for iterating through it
        if self.test:
            if self.holdout:
                return len(self.testIDs)
            else:
                return len(self.valIDs)
        else:
            return len(self.trainIDs)

    def __getitem__(self, idx):
        if self.test:
            if self.holdout:
                #print("test holdout")
                volumes = np.load(os.path.join(self.dataDirectory, self.testIDs[idx] + ".npy"))
            else:
                volumes = np.load(os.path.join(self.dataDirectory, self.valIDs[idx] + ".npy"))
        else:
            volumes = np.load(os.path.join(self.dataDirectory, self.trainIDs[idx] + ".npy"))

        return volumes

    def getValIDs(self):
        return self.valIDs

    def getTestIDs(self):
        return self.testIDs

    def getTrainIDs(self):
        return self.trainIDs


class VolumesFromListCV(Dataset):
    def __init__(self, dataDirectory, patientListDirectory, singleLesionList, valFold, testingHoldoutFold, test=False, holdout=False):
        self.dataDirectory = dataDirectory
        self.test = test
        self.holdout = holdout
        self.trainIDs = []
        self.valIDs = []
        self.testIDs = []

        with open(singleLesionList) as f:
            lines = f.read().splitlines()
            for l in lines:
                self.trainIDs.append(l)

        for i in range(10):
            filePath = os.path.join(patientListDirectory, "foldCV" + str(i) + ".txt")
            with open(filePath) as f:
                lines = f.read().splitlines()
            if i == valFold:
                for l in lines:
                    self.valIDs.append(l)
            elif i == testingHoldoutFold:
                for l in lines:
                    self.testIDs.append(l)
            else: #train
                for l in lines:
                    self.trainIDs.append(l)

    def __len__(self):  # The length of the dataset is important for iterating through it
        if self.test:
            if self.holdout:
                return len(self.testIDs)
            else:
                return len(self.valIDs)
        else:
            return len(self.trainIDs)

    def __getitem__(self, idx):
        if self.test:
            if self.holdout:
                #print("test holdout")
                volumes = np.load(os.path.join(self.dataDirectory, self.testIDs[idx] + ".npy"))
            else:
                volumes = np.load(os.path.join(self.dataDirectory, self.valIDs[idx] + ".npy"))
        else:
            volumes = np.load(os.path.join(self.dataDirectory, self.trainIDs[idx] + ".npy"))

        return volumes

    def getValIDs(self):
        return self.valIDs

    def getTestIDs(self):
        return self.testIDs

    def getTrainIDs(self):
        return self.trainIDs