from pandas import DataFrame
import numpy as np


class DataGenerator:

    def __init__(self, preparedData: DataFrame, lookBack: int, batchSize: int, trainMaxIndex: int, countTests: int):
        self.preparedData = preparedData
        self.lookBack = lookBack
        self.batchSize = batchSize
        self.trainMaxIndex = trainMaxIndex
        self.countTests = countTests

    def getTrainArray(self):
        trainGen = self.getTrainGen()
        trainSteps, valSteps, testSteps = self.getStepsCounts()
        X, y = self.__getArray(trainGen, trainSteps)

        return X, y

    def getValArray(self):
        valGen = self.getValGen()
        trainSteps, valSteps, testSteps = self.getStepsCounts()
        X, y = self.__getArray(valGen, valSteps)

        return X, y

    def getTestArray(self):
        testGen = self.getValGen()
        trainSteps, valSteps, testSteps = self.getStepsCounts()
        X, y = self.__getArray(testGen, testSteps)

        return X, y

    def __getArray(self, generator, countSteps):
        data = self.preparedData
        lookBack = self.lookBack
        X = np.zeros((self.batchSize * countSteps, lookBack, data.shape[-1]))
        y = np.zeros((self.batchSize * countSteps, data.shape[-1]))
        currentRowX = 0
        currentRowY = 0
        for i in range(countSteps):
            samples, targets = next(generator)
            for sampleItem in samples:
                X[currentRowX] = sampleItem
                currentRowX += 1
            for targetItem in targets:
                y[currentRowY] = targetItem
                currentRowY += 1

        return X, y

    def getTrainGen(self):
        shuffle = False
        minIndex = 0
        maxIndex = self.trainMaxIndex
        return self.getGenerator(minIndex, maxIndex, shuffle)

    def getValGen(self):
        shuffle = False
        minIndex = self.trainMaxIndex + 1
        maxIndex = self.trainMaxIndex + self.countTests
        return self.getGenerator(minIndex, maxIndex, shuffle)

    def getTestGen(self):
        shuffle = False
        minIndex = self.trainMaxIndex + self.countTests + 1
        maxIndex = None
        return self.getGenerator(minIndex, maxIndex, shuffle)

    def getStepsCounts(self):
        trainSteps = (self.trainMaxIndex - self.lookBack) // self.batchSize
        valSteps = (self.countTests - self.lookBack) // self.batchSize
        testSteps = (len(self.preparedData) - self.trainMaxIndex - self.countTests - self.lookBack) // self.batchSize

        return trainSteps, valSteps, testSteps

    def getGenerator(self, minIndex, maxIndex, shuffle):
        data = self.preparedData
        lookBack = self.lookBack
        batchSize = self.batchSize
        if maxIndex is None:
            maxIndex = len(data) - 1
        i = minIndex + lookBack
        while 1:
            if shuffle:
                rows = np.random.randint(minIndex + lookBack, maxIndex, size=batchSize)
            else:
                if i + batchSize >= maxIndex:
                    i = minIndex + lookBack
                rows = np.arange(i, min(i + batchSize, maxIndex))
                i += len(rows)

            samples = np.zeros((len(rows), lookBack, data.shape[-1]))
            targets = np.zeros((len(rows), data.shape[-1]))

            for j, row in enumerate(rows):
                indices = range(rows[j] - lookBack, rows[j], 1)
                samples[j] = data.iloc[indices]
                targets[j] = data.iloc[rows[j]]

            yield samples, targets
