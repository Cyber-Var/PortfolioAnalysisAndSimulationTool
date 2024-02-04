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

        # X_train, y_train, X_test = super().split_prediction_sets()

        # lstm_model = Sequential()
        # lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(len(X_train[1]), 1)))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units=50, return_sequences=True))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units=50, return_sequences=True))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units=50, return_sequences=False))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(Dense(units=25))
        # lstm_model.add(Dense(units=1))

        # scaler = MinMaxScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        # y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))

        # lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        # TODO: number of epochs controlled by user:
        # lstm_model.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=100)

        # prediction_scaled = lstm_model.predict(X_test_scaled)
        # prediction = scaler.inverse_transform(prediction_scaled)
        # print(prediction)

        self.final = False
        self.evaluateModel()

        self.final = True
        # self.predict_price()

    def evaluateModel(self):
        Regression.reg = self.createModel()

        all_X_trains, all_X_tests, all_y_trains, all_y_tests = super().evaluateModel()

        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.makeLSTMPrediction(all_X_trains[i], all_y_trains[i], all_X_tests[i])
            all_predictions.extend(predictions)

        print(len(all_predictions))
        print("LSTM Evaluation:")
        return super().calculateEvalMetrics(all_predictions, all_y_tests)

    # TODO: n_estimators chosen by user
    def predict_price(self):
        Regression.reg = self.createModel()
        X_train, y_train, X_test = super().split_prediction_sets()

        prediction = self.makeLSTMPrediction(X_train, y_train, X_test)
        print("LSTM Prediction:", prediction, "\n")
        return prediction

    def makeLSTMPrediction(self, X_train, y_train, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))

        if self.final:
            self.reg.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=100)
        else:
            self.reg.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=10)
        prediction_scaled = self.reg.predict(X_test_scaled)
        prediction = scaler.inverse_transform(prediction_scaled)
        return prediction

    def createModel(self):
        num_features = 12
        if self.hold_duration == "1w":
            num_features = 21
        elif self.hold_duration == "1m":
            num_features = 28

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(num_features, 1)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=True))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=True))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=False))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(units=25))
        lstm_model.add(Dense(units=1))

        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        return lstm_model



