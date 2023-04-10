import scipy.io
import numpy as np
import pandas as pd


class cellData:
    def __init__(self):
        """constructor"""
        self.oxfordDatasetPath = "datasets/oxford-battery-degradation-dataset/"
        self.pmattaiDatasetPath = "datasets/pmattia-dataset/"

    def loadExampleDC(self):
        """loads the exampleDC mat file"""
        self.exampleDCOxford = scipy.io.loadmat(
            self.oxfordDatasetPath + f"ExampleDC_C1.mat"
        )
        self.exampleDCOxford = {
            k: v for k, v in self.exampleDCOxford.items() if k[0] != "_"
        }
        self.exampleDCOxford = pd.DataFrame(
            {k: np.array(v).flatten() for k, v in self.exampleDCOxford.items()}
        )

    def loadFullDataset(self):
        """loads the full oxford battery degrdation dataset"""
        self.fullOxfordDataset = scipy.io.loadmat(
            self.oxfordDatasetPath + "Oxford_Battery_Degradation_Dataset_1.mat"
        )
        self.fullOxfordDataset = {
            k: v for k, v in self.fullOxfordDataset.items() if k[0] != "_"
        }
        self.fullOxfordDataset = pd.DataFrame(
            {k: np.array(v).flatten() for k, v in self.fullOxfordDataset.items()}
        )

    def loadPMAttiaDataset(self):
        """loads the Peter M Attia battery degrdataion dataset"""
        self.pmattiaDataset = pd.read_csv(
            self.pmattaiDatasetPath + "2017-05-12_6C-50per_3_6C_CH36.csv", skipfooter=2
        )

    def plotPMAttiaDataset(self):
        self.pmattiaDataset.plot.line(x="Data_Point", y="Internal_Resistance")
