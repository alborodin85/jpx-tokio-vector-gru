import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from DataExtractor import DataExtractor
from DataPreparer import DataPreparer
import jpx_tokyo_market_prediction

model = load_model('saved-models/trained.h5')
trainMaxIndex = 1100
lookBack = 60

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 400)
np.set_printoptions(linewidth=75, formatter=dict(float=lambda x: "%.3g" % x))

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()
dataExtractor = DataExtractor()
structuredData = dataExtractor.getStructuredData()
structuredData = structuredData[structuredData.index <= '2021-12-03']
dataPreparer = DataPreparer(structuredData, trainMaxIndex)
preparedData = dataPreparer.prepareData()

for (prices, options, financials, trades, secondaryPrices, samplePrediction) in iter_test:

    vocStructuredData = {}
    for securitiesCode in prices['SecuritiesCode'].unique():
        firmSeries = prices[prices['SecuritiesCode'] == securitiesCode]
        securitiesCode = str(securitiesCode)
        date = firmSeries['Date']
        close = firmSeries['Close']
        close.index = date
        vocStructuredData[securitiesCode] = close
        continue

    newRowDf = pd.DataFrame(vocStructuredData)
    structuredData = pd.concat([structuredData, newRowDf], axis=0)
    preparedData = structuredData.pct_change()

    inputDataDf = preparedData.iloc[-lookBack:]
    inputData = inputDataDf.values
    inputData = inputData.reshape(1, inputData.shape[0], inputData.shape[1])

    lastRow = model.predict(inputData)

    lastRow = lastRow[0]
    lastRow = pd.Series(lastRow)
    previousRow = structuredData.iloc[-1]
    restoredRow = dataPreparer.restoreData(lastRow, previousRow)
    restoredRow.index = previousRow.index
    delta = (restoredRow - previousRow) / previousRow
    deltaSorted = delta.sort_values(ascending=False)

    deltas = pd.DataFrame(delta)
    deltas['Rank'] = pd.Series()

    for rankValue in range(deltaSorted.index.size):
        orgId = deltaSorted.index[rankValue]
        deltas['Rank'][orgId] = rankValue

    samplePrediction['Rank'] = deltas['Rank'].values
    samplePrediction['Rank'] = [int(item) for item in samplePrediction['Rank']]

    print(samplePrediction)

    env.predict(samplePrediction)
