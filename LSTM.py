from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

from Regression import Regression


class LSTMAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date, params, is_long, investment_amount):
        super().__init__(hold_duration, data, prediction_date, start_date, is_long)
        self.params = params
        self.investment_amount = investment_amount

    def get_data_for_prediction(self):
        train_start = date.today() - self.historic_date_range
        data_for_prediction = self.data[train_start:]
        return data_for_prediction

    def evaluateModel(self):
        Regression.reg = self.createModel()

        all_X_trains, all_y_trains, all_X_tests, all_y_tests = super().evaluateModel()

        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.make_prediction(all_X_trains[i], all_y_trains[i], all_X_tests[i], False)
            all_predictions.extend(predictions)

        return super().calculateEvalMetrics(all_predictions, all_y_tests)

    # TODO: n_estimators chosen by user
    def predict_price(self, data):
        Regression.reg = self.createModel()

        X_train, y_train, X_test, _ = super().prepareData(data, [], False)

        prediction = self.make_prediction(X_train, y_train, X_test, True)
        return prediction

    def process_features(self, li):
        result = li["Adj Close"].values.tolist()
        return result

    def make_prediction(self, X_train, y_train, X_test, final):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))

        # TODO: number of epochs controlled by user:
        if final:
            self.reg.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=self.params[6])
        else:
            self.reg.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=self.params[6])

        prediction_scaled = self.reg.predict(X_test_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        profit_loss_amount = super().calculate_profit_or_loss(prediction, self.investment_amount)
        return profit_loss_amount

    def createModel(self):
        num_features = 5
        if self.hold_duration == "1w":
            num_features = 20
        elif self.hold_duration == "1m":
            num_features = 80

        # lstm_model = Sequential()
        # lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(num_features, 1)))
        # # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units=50, return_sequences=True))
        # # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units=50, return_sequences=True))
        # # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units=150, return_sequences=False))
        # # lstm_model.add(Dropout(0.2))
        # lstm_model.add(Dense(units=25))
        # lstm_model.add(Dense(units=1))
        #
        # lstm_model.compile(optimizer='adam', loss='mean_squared_error')

        lstm_model = Sequential()
        for _ in range(self.params[0] - 1):
            lstm_model.add(LSTM(units=self.params[1], return_sequences=True, input_shape=(num_features, 1)))
            lstm_model.add(Dropout(self.params[2]))
        lstm_model.add(LSTM(units=self.params[1], return_sequences=False))
        lstm_model.add(Dropout(self.params[2]))
        lstm_model.add(Dense(units=self.params[3]))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer=self.params[4], loss=self.params[5])

        return lstm_model



