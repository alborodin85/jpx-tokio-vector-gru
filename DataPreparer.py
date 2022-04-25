from pandas import DataFrame
from pandas import Series
import pandas as pd
import numpy as np


class DataPreparer:
    normalizedData: DataFrame
    differentiatedData: DataFrame
    structuredData: DataFrame
    stds: Series
    means: Series

    def __init__(self, structuredData: DataFrame, trainMaxIndex):
        self.structuredData = structuredData
        self.trainMaxIndex = trainMaxIndex
        self.stds = Series(dtype='float64')
        self.means = Series(dtype='float64')

    def prepareData(self):
        self.__differentiateData()
        self.__normalizeData()

        return self.normalizedData

    def restoreData(self, predicts: Series, previousRow: Series):
        resultDict = {}
        for orgNumber in predicts.index:
            prevValue = previousRow[orgNumber]
            deviation = predicts[orgNumber]
            # deviation = deviation * self.stds[orgNumber]
            # deviation = deviation + self.means[orgNumber]
            newValue = prevValue * (deviation + 1)
            resultDict[orgNumber] = newValue

        resultSeries = Series(resultDict)
        return resultSeries

    def __differentiateData(self):
        dataDf = pd.DataFrame.copy(self.structuredData)
        dataDf = dataDf.pct_change()
        dataDf = dataDf.fillna(0)
        self.differentiatedData = dataDf

    def __normalizeData(self):
        self.normalizedData = pd.DataFrame.copy(self.differentiatedData)
        return

        mean = self.differentiatedData[:self.trainMaxIndex].mean(axis=0)
        dataDf = self.differentiatedData - mean
        std = self.differentiatedData.std(axis=0, ddof=0)
        dataDf = dataDf / std
        self.normalizedData = dataDf
        self.means = mean
        self.stds = std
