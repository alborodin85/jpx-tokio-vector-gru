from DataPreparer import DataPreparer
from DataExtractor import DataExtractor
from DataGenerator import DataGenerator
from NaivePredictor import NaivePredictor
from GruPredictorGenerator import GruPredictorGenerator
from GruPredictorArray import GruPredictorArray

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 200)

np.set_printoptions(linewidth=75, formatter=dict(float=lambda x: "%.3g" % x))

allRanks = pd.read_csv('all_ranks.csv', index_col=0)
print(allRanks)

resultDf = pd.DataFrame()

for date in allRanks.index:
    currRow = allRanks.loc[date]
    for orgId in currRow.index:
        rankValue = currRow[orgId]
        rowItem = pd.DataFrame({'Date': [date], 'SecuritiesCode': [orgId], 'Rank': [rankValue]})
        resultDf = pd.concat([resultDf, rowItem])

resultDf.to_csv('submission.csv', index=False)
print(resultDf)

exit()
trainMaxIndex = 1100
lookBack = 60
dataExtractor = DataExtractor()
structuredData = dataExtractor.getStructuredData()
rawExtData = dataExtractor.getRawExt()
# print(rawExtData)
# print(structuredData)

dataPreparer = DataPreparer(structuredData, trainMaxIndex)
preparedData = dataPreparer.prepareData()
# print(preparedData)

model = load_model('saved-models/trained.h5')

# batchSize = 25
# dataGenerator = DataGenerator(preparedData=preparedData, lookBack=lookBack, batchSize=batchSize, trainMaxIndex=1100, countTests=158)
# X, y = dataGenerator.getTrainArray()
# X_val, y_val = dataGenerator.getValArray()
# results = model.evaluate(X_val, y_val)
# print(results)

ranksDict = {'Date': rawExtData.Date, 'SecuritiesCode': rawExtData.SecuritiesCode, 'Rank': 0}
ranks = pd.DataFrame(ranksDict, index=rawExtData.index)
# print(ranks)

dates = rawExtData['Date'].unique()

allDeltas = pd.DataFrame()
allRanks = pd.DataFrame()

for currDate in dates:
    endRowIndex, = np.where(preparedData.index.values == currDate)

    inputDataDf = preparedData.iloc[endRowIndex[0] - lookBack:endRowIndex[0]]
    inputData = inputDataDf.values
    inputData = inputData.reshape(1, inputData.shape[0], inputData.shape[1])

    lastRow = model.predict(inputData)
    lastRow = lastRow[0]
    lastRow = pd.Series(lastRow)

    previousRow = structuredData.iloc[endRowIndex[0] - 1]
    lastRow.index = previousRow.index

    restoredRow = dataPreparer.restoreData(lastRow, previousRow)

    # testRow = structuredData.iloc[endRowIndex[0]]
    # print(testRow)
    # print(restoredRow)
    # testRow[:100].plot(style="b")
    # restoredRow[:100].plot(style="r")
    # plt.show()

    delta = (restoredRow - previousRow) / previousRow

    orgDict = {}
    for orgId in restoredRow.index:
        orgDict[orgId] = delta[orgId]

    deltas = pd.DataFrame(orgDict, index=[currDate])

    allDeltas = pd.concat([allDeltas, deltas], axis=0)

    # print(deltas)

    orgsSorted = deltas.T.sort_values(by=currDate, ascending=False)

    # print(orgsSorted)
    # print(orgsSorted.index[3])
    # print(orgsSorted.index.size)
    rankDf = deltas
    for rankValue in range(orgsSorted.index.size):
        orgId = orgsSorted.index[rankValue]
        rankDf[orgId][currDate] = rankValue

    allRanks = pd.concat([allRanks, rankDf], axis=0)

print(allDeltas)
print(allRanks)

allRanks.to_csv('all_ranks.csv')
