from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class Regression:
    reg = None

    def __init__(self, hold_duration, data, prediction_date, start_date):
        self.hold_duration = hold_duration
        self.data = data
        self.prediction_date = prediction_date
        self.start_date = start_date

        self.days = 5
        self.num_days = 0
        if self.hold_duration == "1w":
            self.days = 20
            self.num_days = 4
        elif self.hold_duration == "1m":
            self.days = 80
            self.num_days = 20

    def evaluateModel(self):
        # TODO: explain this sliding method clearly in report

        # all_predictions = []
        all_X_trains = []
        all_X_tests = []
        all_y_trains = []
        all_y_tests = []

        train_start = self.start_date
        counter = 0
        while True:
            if self.hold_duration == "1d":
                train_end = train_start + relativedelta(months=6)
            elif self.hold_duration == "1w":
                train_end = train_start + relativedelta(years=1)
            else:
                train_end = train_start + relativedelta(years=3)

            test_start = train_end + relativedelta(days=1)
            test_end = train_end + relativedelta(months=1)

            if test_end > date.today():
                break

            train = self.data[self.start_date:train_end]
            test = pd.concat([train.tail(self.days + self.num_days), self.data[test_start:test_end]], axis=0)

            X_train, y_train, X_test, y_test = self.prepareData(train, test, True)

            all_X_trains.append(X_train)
            all_X_tests.append(X_test)
            all_y_trains.append(y_train)
            all_y_tests.extend(y_test)

            train_start += relativedelta(months=1)
            counter += 1

        return all_X_trains, all_X_tests, all_y_trains, all_y_tests

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
        self.reg.fit(X_train, y_train)
        prediction = self.reg.predict(X_test)
        return prediction

    def prepareData(self, train_set, test_set, eval):
        X_train = [self.process_features(train_set[i:i + self.days]) for i in range(0, len(train_set) -
                                                                                    (self.days + self.num_days))]
        y_train = train_set["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()

        if eval:
            X_test = [self.process_features(test_set[i:i + self.days]) for i in range(0, len(test_set) -
                                                                                      (self.days + self.num_days))]
            y_test = test_set["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()
        else:
            X_test = [self.process_features(train_set[-self.days:])]
            y_test = []

        return X_train, y_train, X_test, y_test

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

        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}\n')

        return mse, rmse, mae, mape, r2
