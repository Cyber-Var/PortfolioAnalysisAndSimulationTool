import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class Regression:

    reg = None

    def __init__(self, end_date, user_end_date, data, future_data):
        self.end_date = end_date
        self.user_end_date = user_end_date
        self.data = data
        self.future_data = future_data

        self.days = len(self.future_data) + 1
        if self.user_end_date in self.future_data.index:
            self.days -= 1

    def prepareData(self):
        if self.days > 2:
            y = self.data.iloc[(self.days-1)::self.days, :]["Adj Close"].copy()
            y[self.data.index[self.days - 1]] = self.data.iloc[self.days - 1]["Adj Close"]

            X_list = self.data.drop(y.index)["Adj Close"].copy()
            i = self.days - 1
            if i > 1:
                X_list = [X_list.iloc[x:x + i] for x in range(0, len(self.data), i)][:len(y)]
            X = np.array([df.values.flatten() for df in X_list])
            y = y[:len(X_list)]
        else:
            X = self.data.iloc[::2]
            y = self.data.iloc[1::2]["Adj Close"][:len(X)]
            X = X[:len(y)]
        return X, y

    def evaluateModel(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.reg.fit(X_train, y_train)
        predictions = self.reg.predict(X_test)

        # print(y_test)
        # print(rf_predictions)

        return y_test, predictions

    def calculateMetrics(self, y_test, predictions):
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        r2 = r2_score(y_test, predictions)
        return mse, rmse, mae, mape, r2

    def runAlgorithm(self, X, y):
        self.reg.fit(X, y)

        if self.days > 2:
            X_data = self.future_data.tail(self.days)["Adj Close"]
            X_test = [X_data[X_data.index != self.user_end_date]]
        else:
            X_test = self.future_data.head(1)
        # print(X_test)
        print("Actual:", self.future_data.tail(1)["Adj Close"].values[0])

        predictions = self.reg.predict(X_test)
        return predictions
