import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class LinearRegressionAlgorithm:

    def __init__(self, end_date, user_end_date, data, future_data):
        self.end_date = end_date
        self.user_end_date = user_end_date
        self.data = data
        self.future_data = future_data

        self.splitData()

        X, y, X_train, X_test, y_train, y_test = self.splitData()
        mse, rmse, mae, mape, r2 = self.evaluateModel(X_train, X_test, y_train, y_test)
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}')

        lin_predictions = self.runAlgorithm(X, y)

    def splitData(self):
        X = self.data[['Open', 'High', 'Low', 'Volume']]
        y = self.data['Adj Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        return X, y, X_train, X_test, y_train, y_test

    def evaluateModel(self, X_train, X_test, y_train, y_test):
        lin_reg = LinearRegression()
        lin_reg.fit(X_train.values, y_train)

        lin_predictions = lin_reg.predict(X_test.values)

        mse = mean_squared_error(y_test, lin_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, lin_predictions)
        mape = mean_absolute_percentage_error(y_test, lin_predictions) * 100
        r2 = r2_score(y_test, lin_predictions)

        return mse, rmse, mae, mape, r2

    def runAlgorithm(self, X, y):
        lin_reg = LinearRegression()
        lin_reg.fit(X.values, y)

        features = self.future_data.iloc[-1][['Open', 'High', 'Low', 'Volume']].values.reshape(1, -1)
        lin_predictions = lin_reg.predict(features)

        print(lin_predictions)
        print("Actual:", self.future_data.iloc[-1]["Adj Close"])

        return lin_predictions
