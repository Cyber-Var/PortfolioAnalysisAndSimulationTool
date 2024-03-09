from datetime import date
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler


class Regression:
    reg = None

    def __init__(self, hold_duration, data, prediction_date, start_date, is_long):
        self.hold_duration = hold_duration
        self.data = data
        self.prediction_date = prediction_date
        self.start_date = start_date
        self.is_long = is_long

        if self.hold_duration == "1d":
            self.days = 5
            self.num_days = 0
            self.historic_date_range = relativedelta(months=6)
        elif self.hold_duration == "1w":
            self.days = 20
            self.num_days = 4
            self.historic_date_range = relativedelta(years=1)
        else:
            self.days = 80
            self.num_days = 20
            self.historic_date_range = relativedelta(years=3)

    def evaluateModel(self):
        # TODO: explain this sliding method clearly in report

        all_X_trains = []
        all_X_tests = []
        all_y_trains = []
        all_y_tests = []

        train_start = self.start_date
        counter = 0
        while True:
            train_end, test_start, test_end = self.get_train_test_dates(train_start)

            if test_end > date.today():
                break

            train, test = self.split_train_test_sets(train_end, test_start, test_end)

            X_train, y_train, X_test, y_test = self.prepareData(train, test, True)
            all_X_trains.append(X_train)
            all_X_tests.append(X_test)
            all_y_trains.append(y_train)
            all_y_tests.extend(y_test)

            train_start += relativedelta(months=1)
            counter += 1

        return all_X_trains, all_y_trains, all_X_tests, all_y_tests

    def get_train_test_dates(self, train_start):
        train_end = train_start + self.historic_date_range
        test_start = train_end + relativedelta(days=1)
        test_end = train_end + relativedelta(months=1)
        return train_end, test_start, test_end

    def split_train_test_sets(self, train_end, test_start, test_end):
        train = self.data[self.start_date:train_end]
        test = pd.concat([train.tail(self.days + self.num_days), self.data[test_start:test_end]], axis=0)
        return train, test

    def split_prediction_sets(self):
        if self.hold_duration == "1d":
            train_start = date.today() - relativedelta(months=6)
        elif self.hold_duration == "1w":
            train_start = date.today() - relativedelta(years=1)
        else:
            train_start = date.today() - relativedelta(years=3)
        train = self.data[train_start:]

        X_train, y_train, X_test, y_test = self.prepareData(train, [], False)
        return X_train, y_train, X_test

    def makePrediction(self, X_train, y_train, X_test):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
        y_train_scaled_1d = y_train_scaled.ravel()

        X_test_scaled = scaler_X.transform(X_test)
        self.reg.fit(X_train_scaled, y_train_scaled_1d)
        prediction = self.reg.predict(X_test_scaled).reshape(-1, 1)
        return scaler_y.inverse_transform(prediction)

    def prepareData(self, train_set, test_set, evaluate):
        X_train, y_train = self.split_X_y(train_set)

        if evaluate:
            X_test, y_test = self.split_X_y(test_set)
        else:
            X_test = [self.process_features(train_set[-self.days:])]
            y_test = []

        return X_train, y_train, X_test, y_test

    def split_X_y(self, data):
        X = [self.process_features(data[i:i + self.days]) for i in range(0, len(data) - (self.days + self.num_days))]
        y = data["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()
        return X, y

    def process_features(self, li):
        # print(li)

        if self.hold_duration == "1d":
            result = li['Adj Close'].values.tolist()
            # Daily changes (differences):
            changes = li["Adj Close"].diff().dropna()
        elif self.hold_duration == "1w":
            result = np.convolve(li['Adj Close'], np.ones(7) / 7, mode='valid').tolist()
            # Weekly changes (differences):
            changes = li['Adj Close'].rolling(window=6).apply(
                lambda x: (x.iloc[0] - x.iloc[5]))
        else:
            result = np.convolve(li['Adj Close'], np.ones(60) / 60, mode='valid').tolist()
            # Monthly changes (differences):
            changes = li['Adj Close'].rolling(window=21).apply(
                lambda x: (x.iloc[0] - x.iloc[20]))

        result += [changes.mean()]

        result += [li['Adj Close'].mean(), li['Open'].mean(), li['Close'].mean(), li['High'].mean(), li['Low'].mean(),
                   li['Volume'].mean()]
        return result

    def calculateEvalMetrics(self, predictions, y_test):
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        r2 = r2_score(y_test, predictions)
        return mse, rmse, mae, mape, r2

    def printEvaluation(self, mse, rmse, mae, mape, r2):
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}\n')

    def calculate_profit_or_loss(self, predicted_price, investment_amount, confidence=False):
        current_price = self.data.iloc[-1]["Adj Close"]
        if not confidence:
            percentage_change = (predicted_price - current_price) / current_price
        else:
            percentage_change = predicted_price / current_price
        profit_or_loss_value = investment_amount * percentage_change
        if not self.is_long:
            profit_or_loss_value = -profit_or_loss_value
        print(current_price)
        print(predicted_price)
        return profit_or_loss_value


