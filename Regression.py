import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class Regression:
    reg = None

    def __init__(self, hold_duration, data, prediction_date):
        self.hold_duration = hold_duration
        self.data = data
        self.prediction_date = prediction_date

    def temp(self, li):
        print(li)
        result = li['Adj Close'].values.tolist()
        result += [li['Adj Close'].mean(), li['Open'].mean(), li['Close'].mean(), li['High'].mean(), li['Low'].mean(),
                   li['Volume'].mean()]
        return result

    def prepareData(self):

        days = 5
        if self.hold_duration == "1w":
            days = 20
        if self.hold_duration == "1m":
            days = 80

        X_train = [self.temp(self.data[i:i + days]) for i in range(0, len(self.data) - days)]
        y_train = self.data["Adj Close"].iloc[days:].values.tolist()
        X_test = [self.temp(self.data[-days:])]

        return X_train, y_train, X_test

    def evaluateModel(self, X, y):
        # TODO: change the testing to sliding method (and write it clearly in report)
        # TODO: use more than 1 year data (maybe 3). COVID
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.reg.fit(X_train, y_train)
        predictions = self.reg.predict(X_test)

        # for true, pred in zip(y_test, predictions):
        #     print(true, pred, true == pred)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        r2 = r2_score(y_test, predictions)

        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}\n')

        return mse, rmse, mae, mape, r2

    def runAlgorithm(self, X_train, y_train, X_test):
        self.reg.fit(X_train, y_train)

        # print("Actual:", self.future_data.tail(1)["Adj Close"].values[0])

        predictions = self.reg.predict(X_test)
        return predictions
