import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class RandomForestRegressionAlgorithm:

    def __init__(self, end_date, user_end_date, data, future_data):
        self.end_date = end_date
        self.user_end_date = user_end_date
        self.data = data
        self.future_data = future_data

        self.days = len(self.future_data) + 1
        if self.user_end_date in self.future_data.index:
            self.days -= 1

        X, y = self.prepareData()

        mse, rmse, mae, mape, r2 = self.evaluateModel(X, y)
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}')

        rf_predictions = self.runAlgorithm(X, y)
        print("Prediction:", rf_predictions[0])

    def prepareData(self):
        y = self.data.iloc[(self.days-1)::self.days, :]["Adj Close"].copy()
        y[self.data.index[self.days - 1]] = self.data.iloc[self.days - 1]["Adj Close"]

        X_list = self.data.drop(y.index)["Adj Close"].copy()
        i = self.days - 1
        X_list = [X_list.iloc[x:x + i] for x in range(0, len(self.data), i)][:len(y)]
        X = np.array([df.values.flatten() for df in X_list])
        y = y[:len(X_list)]

        return X, y

    def evaluateModel(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_train)
        rf_predictions = rf_reg.predict(X_test)

        # print(y_test)
        # print(rf_predictions)

        mse = mean_squared_error(y_test, rf_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, rf_predictions)
        mape = mean_absolute_percentage_error(y_test, rf_predictions) * 100
        r2 = r2_score(y_test, rf_predictions)

        return mse, rmse, mae, mape, r2

    # TODO: n_estimators chosen by user
    def runAlgorithm(self, X, y):
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X, y)

        X_data = self.future_data.tail(self.days)["Adj Close"]
        X_test = [X_data[X_data.index != self.user_end_date]]
        # print(X_test)
        # print(self.future_data.tail(1))

        rf_predictions = rf_reg.predict(X_test)
        return rf_predictions






