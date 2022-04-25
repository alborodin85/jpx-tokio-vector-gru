import pandas as pd
from pandas import DataFrame
import os.path


class DataExtractor:
    rawExtData: DataFrame
    mainDataFile = 'data/stock_prices.csv'
    extDataFile = 'data_updates/stock_prices.csv'
    structuredDataFile = 'data/data_close.csv'

    def getRawExt(self):
        rawExtData = pd.read_csv(self.extDataFile, index_col=0)
        return rawExtData

    def getStructuredData(self):
        if os.path.exists(self.structuredDataFile):
            structuredData = pd.read_csv(self.structuredDataFile, index_col=0)
            return structuredData

        rawData = pd.read_csv(self.mainDataFile, index_col=0)
        updates = pd.read_csv(self.extDataFile, index_col=0)
        rawData = pd.concat([rawData, updates], axis=0)

        vocStructuredData = {}

        for securitiesCode in rawData['SecuritiesCode'].unique():
            firmSeries = rawData[rawData['SecuritiesCode'] == securitiesCode]
            date = firmSeries['Date']
            close = firmSeries['Close']
            close.index = date
            vocStructuredData[securitiesCode] = close
            continue

        structuredData = DataFrame(vocStructuredData)
        structuredData.to_csv(self.structuredDataFile)

        return structuredData
