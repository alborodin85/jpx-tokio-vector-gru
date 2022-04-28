import pandas as pd
import numpy as np

import jpx_tokyo_market_prediction

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 300)
np.set_printoptions(linewidth=75, formatter=dict(float=lambda x: "%.3g" % x))

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

pricesList = pd.DataFrame()
pricesListDiff = pd.DataFrame()
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
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
    pricesList = pd.concat([pricesList, newRowDf], axis=0)
    if pricesList.index.values.size == 1:
        # тестовые данные
        sample_prediction['Rank'] = [item for item in range(1999, -1, -1)]
        # без этой строки ответ может быть не принят
        sample_prediction['Rank'] = [int(item) for item in sample_prediction['Rank']]
    else:
        pricesListDiff = pricesList.pct_change()
        lastRow = pricesListDiff.iloc[-1]
        lastRow = pd.Series(lastRow)
        deltaSorted = lastRow.sort_values(ascending=False)

        deltas = pd.DataFrame(lastRow)
        deltas['Rank'] = pd.Series()
        for rankValue in range(deltaSorted.index.size):
            orgId = deltaSorted.index[rankValue]
            deltas['Rank'][orgId] = rankValue

        sample_prediction['Rank'] = deltas['Rank'].values

    sample_prediction['Rank'] = [int(item) for item in sample_prediction['Rank']]
    print(sample_prediction)

    env.predict(sample_prediction)
