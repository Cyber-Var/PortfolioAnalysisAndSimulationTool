#
#
#
#
# from Regression import Regression
#
#
# class LSTMAlgorithm(Regression):
#
#     def __init__(self, hold_duration, data, prediction_date):
#         # super().__init__(hold_duration, data, prediction_date)
#
#         X_train, y_train, X_test = super().prepareData()
#
#         # scaler = MinMaxScaler()
#         # scaled_data = scaler.fit_transform(data)
#
#         model = Sequential()
#         model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
#         model.add(Dense(units=1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X_train, y_train, epochs=100, batch_size=32)
#
#         print(model.predict(X_test))

    # def evaluateModel(self, X, y):
    #     Regression.reg = LinearRegression()
    #
    #     print("Linear Regression Evaluation:")
    #     return super().evaluateModel(X, y)
    #
    # # TODO: n_estimators chosen by user
    # def runAlgorithm(self, X_train, y_train, X_test):
    #     Regression.reg = LinearRegression()
    #     predictions = super().runAlgorithm(X_train, y_train, X_test)
    #     print("Linear Regression Prediction:", predictions, "\n")
    #     return predictions
