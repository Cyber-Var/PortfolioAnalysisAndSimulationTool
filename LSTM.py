from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from Regression import Regression


class LSTMAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        days = 5
        if hold_duration == "1w":
            days = 20
        elif hold_duration == "1m":
            days = 80
        num_features = days + 7

        # model = Sequential()
        # model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['close']))))
        # model.add(Dropout(0.3))
        # model.add(LSTM(120, return_sequences=False))
        # model.add(Dropout(0.3))
        # model.add(Dense(20))
        # model.add(Dense(1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(days, num_features)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        X_train, y_train, X_test = super().split_prediction_sets()

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=100)

        test_predict = model.predict(X_test)
        print(test_predict)



