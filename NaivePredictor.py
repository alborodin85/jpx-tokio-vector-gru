import numpy as np
from pandas import Series


class NaivePredictor:
    stds: Series
    means: Series

    def __init__(self, valSteps, valGen, stds, means):
        self.valSteps = valSteps
        self.valGen = valGen
        self.stds = stds
        self.means = means

    def evaluate(self):
        batchMaes = []
        for step in range(self.valSteps):
            samples, targets = next(self.valGen)
            zeroSamplesLine = np.zeros(targets.shape[-1])
            # zeroSamplesLine -= self.means
            # zeroSamplesLine /= self.stds
            zeroSamples = np.zeros(targets.shape)
            for i in range(zeroSamples.shape[0]):
                zeroSamples[i] = zeroSamplesLine
            mae = np.mean(np.abs(zeroSamples - targets))
            batchMaes.append(mae)

        return np.mean(batchMaes)
