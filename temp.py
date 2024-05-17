# import os
#
# folder = 'parameter tuning'
#
# for f in os.listdir(folder):
#     file_path = os.path.join(folder, f)
#     with open(file_path, 'a+') as file:
#         file.write("\n1w, AAPL, 26.02.24\n")

# def setup_model(self, data, auto):
    # if auto:
    #     if self.hold_duration == "1d":
    #         arima_model = auto_arima(data["Adj Close"],
    #                                  seasonal=True,
    #                                  stepwise=True,
    #                                  suppress_warnings=True,
    #                                  error_action="ignore",
    #                                  trace=False,
    #                                  maxiter=self.params[0])
    #         self.arima_order = arima_model.order
    #     elif self.hold_duration == "1w":
    #         arima_model = auto_arima(data["Adj Close"],
    #                                  seasonal=True,
    #                                  stepwise=True,
    #                                  m=5,  # weekly seasonality
    #                                  suppress_warnings=False,
    #                                  error_action="ignore",
    #                                  trace=False,
    #                                  maxiter=self.params[0])
    #         self.arima_order = arima_model.order
    #     else:
    #         arima_model = auto_arima(data["Adj Close"],
    #                                  seasonal=True,
    #                                  stepwise=True,
    #                                  m=12,  # monthly seasonality
    #                                  suppress_warnings=False,
    #                                  error_action="ignore",
    #                                  trace=False,
    #                                  maxiter=self.params[0])
    #         self.arima_order = arima_model.order
    # else:


# def make_prediction(self, model, data):
    # if auto:
    #     if self.hold_duration == "1d":
    #         prediction, confidence_interval = model.predict(n_periods=1, return_conf_int=True)
    #     elif self.hold_duration == "1w":
    #         prediction, confidence_interval = model.predict(n_periods=5, return_conf_int=True)
    #     else:
    #     future_dates = pd.date_range(start=date.today(), end=self.prediction_date,
    #                                  freq='D').map(
    #         lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
    #     prediction, confidence_interval = model.predict(n_periods=len(future_dates) - 1, return_conf_int=True)
    #     return prediction, confidence_interval[-1]
    # else:


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
