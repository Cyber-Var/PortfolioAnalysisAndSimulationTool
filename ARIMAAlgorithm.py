import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
from dateutil.relativedelta import relativedelta

from Regression import Regression


class ARIMAAlgorithm(Regression):
    """
        This class is used for initializing, training and predicting using the ARIMA model.
    """

    def __init__(self, hold_duration, data, prediction_date, start_date, today_data, params, is_long,
                 investment_amount):
        """
        Constructor method
        :param hold_duration: user selected hold duration
        :param data: data fetched for user selected the stock from API
        :param prediction_date: date for which to predict outcomes
        :param start_date: start date of the fetched historical price data
        :param today_data: current price of stock
        :param params: parameters for initializing the model
        :param is_long: user selected short/long investment type
        :param investment_amount: dollar amount invested in this stock
        """
        super().__init__(hold_duration, data, prediction_date, start_date, is_long)
        self.today_data = today_data
        self.params = params
        self.investment_amount = investment_amount

        warnings.filterwarnings("ignore")

        self.start_date = start_date
        self.hold_duration = hold_duration
        self.data = data

        # Set up the (p, d, q) parameters:
        self.arima_order = (self.params[1], self.params[2], self.params[3])

        # Define the number of days to display on the ARIMA's graph:
        if hold_duration == "1d":
            self.historical_range_for_graph = 30
        elif hold_duration == "1w":
            self.historical_range_for_graph = 60
        else:
            self.historical_range_for_graph = 90

    def get_data_for_prediction(self):
        """
        Method for reducing the amount of historical data analyzed:
        :return: dataframe with reduced historical dataset
        """

        # Reduce the amount of historical data analyzed:
        if self.hold_duration == "1d":
            train_start = date.today() - relativedelta(months=6)
        elif self.hold_duration == "1w":
            train_start = date.today() - relativedelta(years=1)
        else:
            train_start = date.today() - relativedelta(years=2)
        data_for_prediction = self.data[train_start:].copy()

        # Return the reduced dataset:
        data_for_prediction['Date'] = data_for_prediction.index
        data_for_prediction.reset_index(drop=True, inplace=True)
        return data_for_prediction

    def predict_price(self, data):
        """
        Method for predicting future stock outcome
        :param data: historical dataset used for prediction
        :return: predicted price
        """

        # Set up the ARIMA model:
        arima_model = self.setup_model(data)
        # Predict price and confidence interval:
        predicted_price, confidence_interval = self.make_prediction(arima_model)

        # Format the predictions:
        prediction = predicted_price.iloc[-1]
        confidence = abs(prediction - confidence_interval[0])

        # Calculate profit/loss amount with confidence interval:
        profit_loss_amount = super().calculate_profit_or_loss(prediction, self.investment_amount)
        profit_loss_confidence = (confidence_interval[0] / prediction) * profit_loss_amount * 0.1
        return profit_loss_amount, predicted_price, confidence, profit_loss_confidence

    def setup_model(self, data):
        """
        Method that initializes the ARIMA model:
        :param data: historical dataset used for training
        :return: the model
        """
        arima_model = ARIMA(data["Adj Close"], order=self.arima_order, enforce_stationarity=False).fit()
        return arima_model

    def make_prediction(self, model):
        """
        Method that predicts future stock price with confidence interval
        :param model: ARIMA model used for predicting
        :return: predicted price and confidence interval
        """

        # Predict price depending on user selected hold duration:
        if self.hold_duration == "1d":
            prediction = model.forecast(steps=1)
            forecast_results = model.get_forecast(steps=1)
        elif self.hold_duration == "1w":
            prediction = model.forecast(steps=5)
            forecast_results = model.get_forecast(steps=5)
        else:
            future_dates = pd.date_range(start=date.today(), end=self.prediction_date,
                                         freq='D').map(
                lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
            prediction = model.forecast(steps=len(future_dates) - 1)
            forecast_results = model.get_forecast(steps=len(future_dates) - 1)
        # Retrieve confidence interval:
        confidence_interval = forecast_results.conf_int(alpha=0.05)
        return prediction, [confidence_interval.iloc[-1, 0], confidence_interval.iloc[-1, 0]]

    def evaluateModel(self):
        """
        Method for evaluating the ARIMA model
        :return: MAPE
        """

        all_predictions = []
        all_tests = []

        # Starting historical date:
        train_start = date.today() - self.historic_date_range - relativedelta(days=1)

        # Loop month by month using the sliding window approach:
        counter = 0
        while True:
            train_end = train_start + self.historic_date_range
            dataset = self.data[train_start:train_end]

            train = dataset.iloc[:-1]
            test = dataset.iloc[-1]

            if counter == 250:
                break

            # Predict price:
            arima_model = self.setup_model(train)
            pred, conf = self.make_prediction(arima_model)

            # Store predicted and actual prices:
            all_predictions.append(pred.iloc[-1])
            all_tests.append(test["Adj Close"])

            train_start -= relativedelta(days=1)
            counter += 1

        # Calculate and return MAPE:
        return super().calculateEvalMetrics(all_predictions, all_tests)

    def plot_arima(self, predictions, data, figure):
        """
        Method for plotting a graph with ARIMA's predicted prices
        :param predictions: ARIMA's predictions
        :param data: historical price data used for plotting
        :param figure: Figure object on which to plot
        :return: the plotted figure
        """
        data.index = data["Date"]

        historical_dates = data.index[-self.historical_range_for_graph:]

        # Process the information to be plotted:

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

        starting_index = predictions.index.min()
        last_adj_close = pd.Series([data.iloc[-1]["Adj Close"]], index=[starting_index - 1])
        y_axis = pd.concat([last_adj_close, predictions])
        # y_axis = np.insert(predictions, 0, data.iloc[-1]["Adj Close"])

        historical_prices = data['Adj Close'][-self.historical_range_for_graph:]

        ax = figure.add_subplot(111)

        # Plot the historical data:
        ax.plot(historical_dates, historical_prices, color='blue', label='Historical Adj Close')

        # plot the predicted data:
        if len(future_dates) > len(y_axis):
            middle_index = len(y_axis) // 2
            y_axis = np.insert(y_axis, middle_index + 1, y_axis[middle_index])
        elif len(y_axis) > len(future_dates):
            y_axis = y_axis[1:]
        ax.plot(future_dates, y_axis, color='red', label='ARIMA Forecast')

        # Graphical view settings:

        ax.set_title('ARIMA Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price')

        if self.hold_duration == "1d":
            ticks = x_axis[::10].union(future_dates)
        else:
            ticks = x_axis[::10].union(pd.Series([today, self.prediction_date]))
        ax.set_xticks(ticks)
        tick_labels = [tick.strftime('%Y-%m-%d') for tick in ticks]
        if self.hold_duration == "1d":
            ax.set_xticklabels(labels=tick_labels, fontsize=9, rotation=300)
        else:
            ax.set_xticklabels(labels=tick_labels, fontsize=9, rotation=330)
        ax.set_ylim(bottom=min(historical_prices.min(), y_axis.min()) * 0.99,
                    top=max(historical_prices.max(), y_axis.max()) * 1.01)
        ax.legend()

        return figure
