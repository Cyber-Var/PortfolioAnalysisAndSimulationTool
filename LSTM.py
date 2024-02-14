import numpy as np
import pandas as pd
from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta

from Regression import Regression


class LSTMAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        self.final = False
        self.evaluateModel()

        self.final = True
        if self.hold_duration == "1d":
            train_start = date.today() - relativedelta(months=6)
        elif self.hold_duration == "1w":
            train_start = date.today() - relativedelta(years=1)
        else:
            train_start = date.today() - relativedelta(years=3)
        data = self.data[train_start:]
        prediction = self.predict_price(data)

    def evaluateModel(self):
        Regression.reg = self.createModel()

        all_X_trains, all_X_tests, all_y_trains, all_y_tests = self.prepareForEval()

        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.make_prediction(all_X_trains[i], all_y_trains[i], all_X_tests[i])
            all_predictions.extend(predictions)

        print(len(all_predictions))
        print(all_predictions)
        print(all_y_tests)
        print("LSTM Evaluation:")
        return super().calculateEvalMetrics(all_predictions, all_y_tests)

    def prepareForEval(self):
        # TODO: explain this sliding method clearly in report

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

            X_train = [self.process_features(train[i:i + self.days]) for i in range(0, len(train) -
                                                                                        (self.days + self.num_days))]
            y_train = train["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()

            X_test = [self.process_features(test[i:i + self.days]) for i in range(0, len(test) -
                                                                                  (self.days + self.num_days))]
            y_test = test["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()

            all_X_trains.append(X_train)
            all_X_tests.append(X_test)
            all_y_trains.append(y_train)
            all_y_tests.extend(y_test)

            train_start += relativedelta(months=1)
            counter += 1

        return all_X_trains, all_X_tests, all_y_trains, all_y_tests

    # TODO: n_estimators chosen by user
    def predict_price(self, data):
        Regression.reg = self.createModel()

        X_train = [self.process_features(data[i:i + self.days]) for i in range(0, len(data) -
                                                                                    (self.days + self.num_days))]
        y_train = data["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()
        X_test = [self.process_features(data[-self.days:])]

        prediction = self.make_prediction(X_train, y_train, X_test)
        print("LSTM Prediction:", prediction, "\n")
        return prediction

    def process_features(self, li):
        result = li["Adj Close"].values.tolist()
        return result

    def make_prediction(self, X_train, y_train, X_test):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))

        # TODO: number of epochs controlled by user:
        if self.final:
            self.reg.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=10)
        else:
            self.reg.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=10)
        prediction_scaled = self.reg.predict(X_test_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        return prediction

    def createModel(self):
        num_features = 5
        if self.hold_duration == "1w":
            num_features = 20
        elif self.hold_duration == "1m":
            num_features = 80

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(num_features, 1)))
        # lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=True))
        # lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=True))
        # lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=False))
        # lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(units=25))
        lstm_model.add(Dense(units=1))

        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        return lstm_model



