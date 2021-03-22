## 1.   Reading dataset

import pandas as pd
from datetime import datetime

stocks = pd.read_csv('sphist.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks.sort_values(by=['Date'], ascending=True, inplace=True)

## 2.   Generating indicators

print('Generating indicators...')
shifted_stocks = stocks.shift(1).copy()
shifted_stocks['std_5'] = shifted_stocks['Close'].rolling(5).std()
shifted_stocks['std_365'] = shifted_stocks['Close'].rolling(365).std()
shifted_stocks['std_ratio'] = shifted_stocks['std_5'] / shifted_stocks['std_365']
shifted_stocks['mean_day_5'] = shifted_stocks['Close'].rolling(5).mean()
shifted_stocks['mean_day_30'] = shifted_stocks['Close'].rolling(30).mean()
shifted_stocks['mean_day_365'] = shifted_stocks['Close'].rolling(365).mean()
shifted_stocks['mean_ratio'] = shifted_stocks['mean_day_5'] / shifted_stocks['mean_day_365']
print('Done')

## 3.   Splitting up the data

print('Creating train & test datasets...')
shifted_stocks = shifted_stocks[shifted_stocks['Date'] > datetime(year=1951, month=1, day=2)]
shifted_stocks.dropna(axis=0, inplace=True)

train = shifted_stocks[shifted_stocks['Date'] < datetime(year=2013, month=1, day=1)]
test = shifted_stocks[shifted_stocks['Date'] >= datetime(year=2013, month=1, day=1)]
print('Done')

## 4.   Making predictions

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Train
lr = LinearRegression()
X_train = train[['std_5', 'std_365', 'std_ratio', 'mean_day_5', 'mean_day_30', 'mean_day_365', 'mean_ratio']]
y_train = train['Close']
print('Training...')
lr.fit(X_train, y_train)
print('Done')

# Test
X_test = test[['std_5', 'std_365', 'std_ratio', 'mean_day_5', 'mean_day_30', 'mean_day_365', 'mean_ratio']]
y_test = test['Close']
print('Testing...')
predictions = lr.predict(X_test)
print('Done')

# Metrics
MAE = mean_absolute_error(y_true=y_test, y_pred=predictions)
print(MAE)

"""
5.   Other ideas to improve metric 

    The average volume over the past five days.
    The average volume over the past year.
    The ratio between the average volume for the past five days, and the average volume for the past year.
    The standard deviation of the average volume over the past five days.
    The standard deviation of the average volume over the past year.
    The ratio between the standard deviation of the average volume for the past five days, and the standard deviation of the average volume for the past year.
    The year component of the date.
    The ratio between the lowest price in the past year and the current price.
    The ratio between the highest price in the past year and the current price.
    The month component of the date.
    The day of week.
    The day component of the date.
    The number of holidays in the prior month.
"""