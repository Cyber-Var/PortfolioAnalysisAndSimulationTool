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

    def prepareData(self):
        X_test = self.data.tail(1)
        self.data = self.data.drop(self.data.index[-1])

        if self.hold_duration == "1d":
            X_train = self.data.drop(self.data.index[-1])
            y_train = self.data.drop(self.data.index[0])["Adj Close"]

        else:
            # day_of_week = self.prediction_date.weekday()
            # data_for_day_of_week = self.data[self.data.index.weekday == 0]

            days = len(pd.date_range(X_test.index[0] + relativedelta(days=1), self.prediction_date,
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna())

            X_train = self.data.drop(self.data.index[-days:])
            y_train = self.data.drop(self.data.index[:days])["Adj Close"]

        return X_train, y_train, X_test

    def evaluateModel(self, X, y):
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
