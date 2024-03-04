import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

from Regression import Regression


class ARIMAAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date, today_data, params):
        super().__init__(hold_duration, data, prediction_date, start_date)
        self.today_data = today_data
        self.params = params

        warnings.filterwarnings("ignore")

        self.start_date = start_date
        self.hold_duration = hold_duration
        self.data = data

        self.arima_order = (self.params[1], self.params[2], self.params[3])

        if hold_duration == "1d":
            self.historical_range_for_graph = 30
        elif hold_duration == "1w":
            self.historical_range_for_graph = 60
        else:
            self.historical_range_for_graph = 90

    def get_data_for_prediction(self):
        if self.hold_duration == "1d":
            train_start = date.today() - relativedelta(months=6)
        elif self.hold_duration == "1w":
            train_start = date.today() - relativedelta(years=1)
        else:
            train_start = date.today() - relativedelta(years=2)
        data_for_prediction = self.data[train_start:].copy()

        data_for_prediction['Date'] = data_for_prediction.index
        data_for_prediction.reset_index(drop=True, inplace=True)
        return data_for_prediction

    def predict_price(self, data):
        arima_model = self.setup_model(data, True)
        predicted_price, confidence_interval = self.make_prediction(arima_model, True)

        # print("ARIMA Prediction:")
        # print(f"Tomorrow's predicted closing price: {predicted_price.iloc[-1]}")
        # print(f"95% confidence interval: between {confidence_interval[0]} and {confidence_interval[1]} \n")

        return predicted_price

    def setup_model(self, data, auto):
        # TODO: remove auto ?
        if auto:
            if self.hold_duration == "1d":
                arima_model = auto_arima(data["Adj Close"],
                                         seasonal=True,  # TODO: try false also
                                         stepwise=True,
                                         suppress_warnings=True,
                                         error_action="ignore",
                                         trace=False,
                                         maxiter=self.params[0])
                self.arima_order = arima_model.order
            elif self.hold_duration == "1w":
                arima_model = auto_arima(data["Adj Close"],
                                         seasonal=True,  # TODO: try false also
                                         stepwise=True,
                                         m=5,  # weekly seasonality
                                         suppress_warnings=False,
                                         error_action="ignore",
                                         trace=False,
                                         maxiter=self.params[0])
                self.arima_order = arima_model.order
            else:
                arima_model = auto_arima(data["Adj Close"],
                                         seasonal=True,
                                         stepwise=True,
                                         m=12,  # monthly seasonality
                                         suppress_warnings=False,
                                         error_action="ignore",
                                         trace=False,
                                         maxiter=self.params[0])
                self.arima_order = arima_model.order
        else:
            arima_model = ARIMA(data["Adj Close"], order=self.arima_order).fit()

        return arima_model

    def make_prediction(self, model, auto):
        if auto:
            if self.hold_duration == "1d":
                prediction, confidence_interval = model.predict(n_periods=1, return_conf_int=True)
            elif self.hold_duration == "1w":
                prediction, confidence_interval = model.predict(n_periods=5, return_conf_int=True)
            else:
                future_dates = pd.date_range(start=date.today(), end=self.prediction_date,
                                             freq='D').map(
                    lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
                prediction, confidence_interval = model.predict(n_periods=len(future_dates) - 1, return_conf_int=True)
        else:
            if self.hold_duration == "1d":
                prediction = model.forecast(steps=1)
            elif self.hold_duration == "1w":
                prediction = model.forecast(steps=5)
            else:
                prediction = model.forecast(steps=20)
            confidence_interval = [[0, 0]]

        # predicted_price = prediction.iloc[-1]
        return prediction, confidence_interval[-1]

    def evaluateModel(self):
        # TODO: explain this sliding method clearly in report

        all_predictions = []
        all_tests = []

        train_start = date.today() - self.historic_date_range - relativedelta(days=1)

        counter = 0
        while True:
            train_end = train_start + self.historic_date_range
            dataset = self.data[train_start:train_end]

            train = dataset.iloc[:-1]
            test = dataset.iloc[-1]

            if counter == 250:
                break

            arima_model = self.setup_model(train, False)
            pred, conf = self.make_prediction(arima_model, False)

            all_predictions.append(pred.iloc[-1])
            all_tests.append(test["Adj Close"])

            train_start -= relativedelta(days=1)
            counter += 1

        return super().calculateEvalMetrics(all_predictions, all_tests)

    def plot_arima(self, predictions, data):
        data.index = data["Date"]

        historical_dates = data.index[-self.historical_range_for_graph:]

        today = date.today()
        future_dates = pd.date_range(start=today, end=self.prediction_date,
                               freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()

        if today.weekday() >= 5:
            future_dates = pd.concat([pd.Series([today]), pd.Series(future_dates)], ignore_index=True)
            date_for_insert = pd.to_datetime(today)
            data.loc[date_for_insert, "Adj Close"] = data.iloc[-1]["Adj Close"]
            historical_dates = historical_dates.union(future_dates[:1])[1:]
        else:
            if self.today_data.index != today:
                data.loc[today, "Adj Close"] = data.iloc[-1]["Adj Close"]
                historical_dates = historical_dates.append(pd.Index([pd.Timestamp(today)]))
                historical_dates = historical_dates[1:]

        x_axis = historical_dates.union(future_dates[:1])
        y_axis = np.insert(predictions, 0, data.iloc[-1]["Adj Close"])

        historical_prices = data['Adj Close'][-self.historical_range_for_graph:]

        plt.plot(historical_dates, historical_prices, color='blue', label='Historical Adj Close')
        plt.plot(future_dates, y_axis, color='red', label='ARIMA Forecast')

        plt.title('ARIMA Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')

        if self.hold_duration == "1d":
            ticks = x_axis[::10].union(future_dates)
        else:
            ticks = x_axis[::10].union(pd.Series([today, self.prediction_date]))
        plt.xticks(ticks, fontsize=9, rotation=330)
        plt.ylim(bottom=min(historical_prices.min(), y_axis.min()) * 0.99,
                 top=max(historical_prices.max(), y_axis.max()) * 1.01)

        plt.legend()
        plt.show()
