import pandas as pd
import numpy as np
import jpx_tokyo_market_prediction


def createClose(prices):
    vocStructuredData = {}
    for securitiesCode in prices['SecuritiesCode'].unique():
        firmSeries = prices[prices['SecuritiesCode'] == securitiesCode]
        securitiesCode = str(securitiesCode)
        date = firmSeries['Date']
        close = firmSeries['Close']
        close.index = date
        vocStructuredData[securitiesCode] = close
        continue

    return pd.DataFrame(vocStructuredData)


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

pricesList = pd.DataFrame()
pricesListDiff = pd.DataFrame()
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    newRowDf = createClose(prices)
    pricesList = pd.concat([pricesList, newRowDf], axis=0)

    if pricesList.index.values.size == 1:
        sample_prediction['Rank'] = [item for item in range(1999, -1, -1)]
    else:
        pricesListDiff = pricesList.pct_change()
        lastRow = pricesListDiff.iloc[-1]

        sample_prediction['Target'] = lastRow.values
        sample_prediction = sample_prediction.sort_values(by="Target", ascending=False)
        sample_prediction.Rank = np.arange(0, 2000)
        sample_prediction = sample_prediction.sort_values(by="SecuritiesCode", ascending=True)
        sample_prediction = sample_prediction[["Date", "SecuritiesCode", "Rank"]]

    sample_prediction['Rank'] = [int(item) for item in sample_prediction['Rank']]
    print(sample_prediction)

    env.predict(sample_prediction)
