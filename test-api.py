import pandas as pd
import numpy as np

import jpx_tokyo_market_prediction
'''Для работы АПИ необходимо скопировать в папку example_test_files в этом проекте файлы из папки example_test_files, предоставляемые организаторами.'''

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 300)
np.set_printoptions(linewidth=75, formatter=dict(float=lambda x: "%.3g" % x))

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    print(prices)
    print(options)
    print(financials)
    print(trades)
    print(secondary_prices)
    print(sample_prediction)
    sample_prediction['Rank'] = 0
    print(sample_prediction)

    env.predict(sample_prediction)
