import sys
if r'C:\borodin_admin\Институт\4-й семестр\Факультатив NLP\Kaggle\jpx-tokio-vector-gru' not in sys.path:
    sys.path.append(r'C:\borodin_admin\Институт\4-й семестр\Факультатив NLP\Kaggle\jpx-tokio-vector-gru')

import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

trgts = {}
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    print(prices)
    print(sample_prediction)
    sample_prediction['Rank'] = 0
    env.predict(sample_prediction)
