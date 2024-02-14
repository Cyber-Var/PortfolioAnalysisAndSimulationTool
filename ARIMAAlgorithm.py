import warnings

import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class ARIMAAlgorithm:

    def __init__(self, data, hold_duration, start_date):
        self.params = None

        if hold_duration == "1m":
            print("ARIMA model not applicable to monthly predictions.")
        else:
            warnings.filterwarnings("ignore")

            self.start_date = start_date
            self.hold_duration = hold_duration
            self.data = data

            if hold_duration == "1d":
                train_start = date.today() - relativedelta(months=6)
            else:
                train_start = date.today() - relativedelta(years=1)
            data_for_prediction = data[train_start:].copy()

            data_for_prediction['Date'] = data_for_prediction.index
            data_for_prediction.reset_index(drop=True, inplace=True)

            self.params = self.predict_price(data_for_prediction)

            self.evaluateModel()

    def predict_price(self, data):
        arima_model = self.setup_model(data, True)
        predicted_price, confidence_interval = self.make_prediction(arima_model, True)

        print("ARIMA Prediction:")
        print(f"Tomorrow's predicted closing price: {predicted_price}")
        print(f"95% confidence interval: between {confidence_interval[0]} and {confidence_interval[1]} \n")

        return arima_model.order

    def setup_model(self, data, auto):
        # data.index = pd.to_datetime(data.index)

        if auto:
            if self.hold_duration == "1d":
                arima_model = auto_arima(data["Adj Close"],
                                         seasonal=True,  # TODO: try false also
                                         stepwise=True,
                                         suppress_warnings=True,
                                         error_action="ignore",
                                         # max_order=None,
                                         trace=False)
            else:
                arima_model = auto_arima(data["Adj Close"],
                                         seasonal=True,  # TODO: try false also
                                         stepwise=True,
                                         m=5,
                                         suppress_warnings=False,
                                         error_action="ignore",
                                         # max_order=None,
                                         trace=False)
        else:
            arima_model = ARIMA(data["Adj Close"], order=(1, 1, 1)).fit()

        # print(arima_model.summary())
        return arima_model

    def make_prediction(self, model, auto):
        if auto:
            if self.hold_duration == "1d":
                prediction, confidence_interval = model.predict(n_periods=1, return_conf_int=True)
            else:
                prediction, confidence_interval = model.predict(n_periods=5, return_conf_int=True)
        else:
            if self.hold_duration == "1d":
                prediction = model.forecast(steps=1)
            else:
                prediction = model.forecast(steps=5)
            confidence_interval = [[0, 0]]

        predicted_price = prediction.iloc[-1]
        return predicted_price, confidence_interval[-1]

    def evaluateModel(self):
        # TODO: explain this sliding method clearly in report

        all_predictions = []
        all_tests = []

        threshold = self.data.iloc[-250:].index[0]
        today = date.today()

        if self.hold_duration == "1d":
            historic_date_range = relativedelta(months=6)
        elif self.hold_duration == "1d":
            historic_date_range = relativedelta(years=1)
        else:
            historic_date_range = relativedelta(years=3)

        train_start = today - historic_date_range - relativedelta(days=1)

        counter = 0
        while True:
            train_end = train_start + historic_date_range
            dataset = self.data[train_start:train_end]

            train = dataset.iloc[:-1]
            test = dataset.iloc[-1]

            if counter == 250:
                break

            arima_model = self.setup_model(train, False)
            pred, conf = self.make_prediction(arima_model, False)

            all_predictions.append(pred)
            all_tests.append(test["Adj Close"])

            train_start -= relativedelta(days=1)
            counter += 1

        mse = mean_squared_error(all_tests, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_tests, all_predictions)
        mape = mean_absolute_percentage_error(all_tests, all_predictions) * 100
        r2 = r2_score(all_tests, all_predictions)

        print("ARIMA Evaluation:")
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}\n')
