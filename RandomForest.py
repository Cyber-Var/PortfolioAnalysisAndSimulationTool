import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class RandomForestRegressionAlgorithm:

    def __init__(self, end_date, user_end_date, data, future_data):
        self.end_date = end_date
        self.user_end_date = user_end_date
        self.data = data
        self.future_data = future_data

        self.data = self.prepareData(self.data)

        X, y, X_train, X_test, y_train, y_test = self.splitData()
        mse, rmse, mae, mape, r2 = self.evaluateModel(X_train, X_test, y_train, y_test)
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}')

        rf_predictions = self.runAlgorithm(X, y)

    def prepareData(self, data):
        preparedData = data.copy()

        if len(data) <= 7:
            rolling_value = 2
        elif len(data) < 20:
            rolling_value = 3
        elif len(data) < 30:
            rolling_value = 4
        else:
            rolling_value = 5

        preparedData["Percent-Change"] = data["Adj Close"].pct_change()
        preparedData["Open-Close-Change"] = (data["Open"] - data["Close"]) / data["Open"]
        preparedData["High-Low-Change"] = (data["High"] - data["Low"]) / data["Low"]
        preparedData["Rolling-Standard-Deviation"] = preparedData["Percent-Change"].rolling(rolling_value).std()
        preparedData["Rolling-Mean"] = preparedData["Percent-Change"].rolling(rolling_value).mean()
        preparedData.dropna(inplace=True)

        return preparedData

    def splitData(self):
        X = self.data[["Open-Close-Change", "High-Low-Change", "Rolling-Standard-Deviation", "Rolling-Mean"]]
        y = self.data['Adj Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        return X, y, X_train, X_test, y_train, y_test

    def evaluateModel(self, X_train, X_test, y_train, y_test):
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train.values, y_train)
        rf_predictions = rf_reg.predict(X_test.values)

        mse = mean_squared_error(y_test, rf_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, rf_predictions)
        mape = mean_absolute_percentage_error(y_test, rf_predictions) * 100
        r2 = r2_score(y_test, rf_predictions)

        return mse, rmse, mae, mape, r2

    def runAlgorithm(self, X, y):
        # TODO: change this

        X = X[:(len(self.data) - 1)]
        y = y[:(len(self.data) - 1)]

        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X.values, y)

        # prediction_dates = pd.date_range(self.end_date, self.user_end_date, freq='B')
        # future_data = pd.DataFrame(index=prediction_dates, columns=self.data.columns)

        # prediction_data = self.data.loc['2023-08-30':'2023-09-20']
        # print(prediction_data)

        # X_test = self.data[["Open-Close-Change", "High-Low-Change", "Rolling-Standard-Deviation", "Rolling-Mean"]].tail(1)
        # rf_predictions = rf_reg.predict(X_test.values)
        #
        # print(rf_predictions)
        # print(self.data[["Adj Close"]].tail(1).values[0][0])
        # print(self.data[["Adj Close"]].head(1).values[0][0])

        rf_predictions = []
        data = self.prepareData(self.future_data)
        if len(data) > 0:
            X_test = data[["Open-Close-Change", "High-Low-Change", "Rolling-Standard-Deviation", "Rolling-Mean"]]
            rf_predictions = rf_reg.predict(X_test.values)
            print(rf_predictions)
            print("Actual:", self.future_data[["Adj Close"]].tail(1).values[0][0])
        else:
            print("Impossible")

        return rf_predictions






