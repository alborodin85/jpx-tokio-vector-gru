import pandas as pd
import numpy as np
import jpx_tokyo_market_prediction

prices = pd.read_csv("data_updates/stock_prices.csv")
average = pd.DataFrame(prices.groupby("SecuritiesCode").Target.mean())


def get_avg(_id_):
    return average.loc[_id_]


prices["Avg"] = prices["SecuritiesCode"].apply(get_avg)

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    ds = [prices, options, financials, trades, secondary_prices, sample_prediction]
    sample_prediction["Avg"] = sample_prediction["SecuritiesCode"].apply(get_avg)
    df = sample_prediction[["Date", "SecuritiesCode", "Avg"]]
    df.Date = pd.to_datetime(df.Date)
    df['Date'] = df['Date'].dt.strftime("%Y%m%d").astype(int)
    sample_prediction["Prediction"] = 0
    sample_prediction = sample_prediction.sort_values(by="Prediction", ascending=False)
    sample_prediction.Rank = np.arange(0, 2000)
    sample_prediction = sample_prediction.sort_values(by="SecuritiesCode", ascending=True)
    sample_prediction.drop(["Prediction"], axis=1)
    submission = sample_prediction[["Date", "SecuritiesCode", "Rank"]]
    print(sample_prediction)
    env.predict(submission)
