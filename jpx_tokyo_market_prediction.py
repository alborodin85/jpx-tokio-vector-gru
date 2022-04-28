import pandas as pd


class make_env:
    pricesPath = 'example_test_files/stock_prices.csv'
    optionsPath = 'example_test_files/options.csv'
    financialPath = 'example_test_files/financials.csv'
    tradesPath = 'example_test_files/trades.csv'
    secondaryPricesPath = 'example_test_files/secondary_stock_prices.csv'
    samplePredictionPath = 'example_test_files/sample_submission.csv'

    def __init__(self):
        self.samplePrediction = pd.DataFrame()

    def iter_test(self):
        allPrices = pd.read_csv(self.pricesPath)
        allOptions = pd.read_csv(self.optionsPath)
        allFinancials = pd.read_csv(self.financialPath)
        allTrades = pd.read_csv(self.tradesPath)
        allSecondaryPrices = pd.read_csv(self.secondaryPricesPath)
        allSamplePrediction = pd.read_csv(self.samplePredictionPath)

        dates = allPrices['Date'].unique()
        for currDate in dates:
            prices = allPrices[allPrices['Date'] == currDate]
            prices.index = range(prices['Date'].count())

            options = allOptions[allOptions['Date'] == currDate]
            options.index = range(options['Date'].count())

            financials = allFinancials[allFinancials['Date'] == currDate]
            financials.index = range(financials['Date'].count())

            trades = allTrades[allTrades['Date'] == currDate]
            trades.index = range(trades['Date'].count())

            secondaryPrices = allSecondaryPrices[allSecondaryPrices['Date'] == currDate]
            secondaryPrices.index = range(secondaryPrices['Date'].count())

            samplePrediction = allSamplePrediction[allSamplePrediction['Date'] == currDate]
            samplePrediction.index = range(samplePrediction['Date'].count())

            yield prices, options, financials, trades, secondaryPrices, samplePrediction

    def predict(self, samplePrediction):
        self.samplePrediction = pd.concat([self.samplePrediction, samplePrediction], axis=0)
        self.samplePrediction.to_csv('submission.csv', index=False)
