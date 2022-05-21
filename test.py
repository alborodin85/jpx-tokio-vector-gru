import datetime

import pandas as pd
import statistics

pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 3)
pd.set_option('display.width', 200)

import os

df = pd.DataFrame({'Test1': [1, 2, 3, 4, 5], 'Test2': [6, 8, 10, 12, 14]})
df.plot()

# print(f'mean of [1, 2, 3, 4]: {statistics.mean([1, 2, 3, 4]):.3f}')
# print(f'mean of [6, 8, 10, 12]: {statistics.mean([6, 8, 10, 12]):.3f}')
# print(f'std of [1, 2, 3, 4]: {statistics.pstdev([1, 2, 3, 4]):.3f}')
# print(f'std of [6, 8, 10, 12]: {statistics.pstdev([6, 8, 10, 12]):.3f}')

mean = df[:4].mean(axis=0)
# print(mean)

std = df[:4].std(axis=0, ddof=0)
# print(std)

normalizedDf = pd.DataFrame.copy(df)
normalizedDf -= mean
# print(normalizedDf)

normalizedDf /= std
# print(normalizedDf)

normalizedDf *= std
normalizedDf += mean

# print(normalizedDf)

differentiatedData = df.pct_change()
differentiatedData = differentiatedData.fillna(0)

mean = differentiatedData[:4].mean(axis=0)
normalizedData = differentiatedData - mean
std = differentiatedData.std(axis=0, ddof=0)
normalizedData = normalizedData / std

column = 1
resultRow = normalizedData.iloc[:, [column]]
meanOne = mean[column]
stdOne = std[column]

resultRow = resultRow * stdOne
resultRow = resultRow + meanOne

initRow = df.iloc[:, [column]]

resultRow.iat[0, 0] = initRow.iloc[0]
for lineNumber in range(1, resultRow.index.values.size):
    prevValue = initRow.iat[lineNumber - 1, 0]
    deviation = resultRow.iat[lineNumber, 0]
    newValue = prevValue * (deviation + 1)
    resultRow.iat[lineNumber, 0] = newValue
    # print(newValue)

print(initRow)
print(resultRow)
# print([mean, meanOne])
# print([std, stdOne])
# print(df)
# print(differentiatedData)
# print(normalizedData)
