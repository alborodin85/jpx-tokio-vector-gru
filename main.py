from DataPreparer import DataPreparer
from DataExtractor import DataExtractor
from DataGenerator import DataGenerator
from NaivePredictor import NaivePredictor
from GruPredictorGenerator import GruPredictorGenerator
from GruPredictorArray import GruPredictorArray

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 200)

np.set_printoptions(linewidth=75, formatter=dict(float=lambda x: "%.3g" % x))

trainMaxIndex = 1100
countTests = 158
lookBack = 60
dataExtractor = DataExtractor()
structuredData = dataExtractor.getStructuredData()
# print(structuredData)

dataPreparer = DataPreparer(structuredData, trainMaxIndex)
preparedData = dataPreparer.prepareData()
# print(preparedData)

# lastRow = preparedData.iloc[1257]
# previousRow = structuredData.iloc[1256]
# restoredRow = dataPreparer.restoreData(lastRow, previousRow)
# print(structuredData.iloc[1257])
# print(restoredRow)

batchSize = 25
dataGenerator = DataGenerator(preparedData=preparedData, lookBack=lookBack, batchSize=batchSize, trainMaxIndex=trainMaxIndex, countTests=countTests)
X, y = dataGenerator.getTrainArray()
X_val, y_val = dataGenerator.getValArray()

# trainGen = dataGenerator.getTrainGen()
valGen = dataGenerator.getValGen()
# testGen = dataGenerator.getTestGen()
trainSteps, valSteps, testSteps = dataGenerator.getStepsCounts()

predictor = NaivePredictor(valSteps, valGen, dataPreparer.stds, dataPreparer.means)
maeNaive = predictor.evaluate()
print(f'MAE naive: {maeNaive:.5f}')
valGen = dataGenerator.getValGen()

numEpochs = 10

# predictor = GruPredictorGenerator()
# history = predictor.makeModelAndFit(preparedData, trainGen, valGen, valSteps, numEpochs, stepsPerEpoch=500)

predictor = GruPredictorArray()
history = predictor.makeModelAndFit(preparedData, X, y, X_val, y_val, numEpochs, batchSize)

loss = history.history['loss']
valLoss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, valLoss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

print('\n')
print(f'MAE naive: {maeNaive:.5f}')
print(f'MAE Sequential: {min(valLoss):.5f}')

plt.show()
