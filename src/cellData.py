import scipy.io
import sys
import os


class cellData:
    def __init__(self):
        """constructor"""
        self.datasetPath = "oxford-battery-degradation-dataset/"

    def loadExampleDC(self):
        """loads the exampleDC mat file"""
        self.exampleDC = scipy.io.loadmat(self.datasetPath + f"ExampleDC_C1.mat")

    def loadFullDataset(self):
        """loads the full oxford battery degrdation dataset"""
        self.fullDataset = scipy.io.loadmat(
            self.datasetPath + "Oxford_Battery_Degradation_Dataset_1.mat"
        )
