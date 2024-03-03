import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout
import sys

from ARIMAAlgorithm import ARIMAAlgorithm
from BayesianRegressionAlgorithm import BayesianRegressionAlgorithm
from MonteCarlo import MonteCarloSimulation
from ParameterTester import ParameterTester
from RiskMetrics import RiskMetrics
from RandomForest import RandomForestAlgorithm
from LinearRegressionAlgorithm import LinearRegressionAlgorithm
from EthicalScore import ESGScores
from LSTM import LSTMAlgorithm
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage
from UI.PortfolioPage import PortfolioPage

import logging


class Controller:

    def __init__(self, hold_duration):
        self.tickers = []
        self.investments = []
        self.hold_duration = hold_duration

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        today = date.today()
        self.end_date = today

        # TODO: choose historical date range here using user's preferred investment behaviour:
        if hold_duration == "1d":
            self.start_date = today - relativedelta(months=18)
            self.prediction_date = today + relativedelta(days=1)
            while self.prediction_date.weekday() >= 5:
                self.prediction_date = self.prediction_date + relativedelta(days=1)
            self.moving_avg_value = 5
        elif hold_duration == "1w":
            self.start_date = today - relativedelta(years=2)
            self.prediction_date = today + relativedelta(days=7)
            self.moving_avg_value = 20
        else:
            self.start_date = today - relativedelta(years=4)
            self.prediction_date = today + relativedelta(months=1)
            self.moving_avg_value = 80

        self.data = []
        # apple_data = self.getDataForTicker("AAPL", self.data)

    def add_ticker(self, ticker, investment):
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            self.investments.append(investment)
            # Retrieve historical data from Yahoo! Finance:
            self.data = self.downloadData(self.start_date, self.end_date)

    def remove_ticker(self, ticker):
        index = self.tickers.index(ticker)
        self.tickers.remove(ticker)
        self.investments.pop(index)
        self.data.drop(ticker, axis=1, inplace=True)

    def run_linear_regression(self, ticker):
        # Linear Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data)
        linear_regression = LinearRegressionAlgorithm(self.hold_duration, data, self.prediction_date,
                                                      self.start_date, (True,))
        return self.run_model(linear_regression, "Linear Regression")

    def run_random_forest(self, ticker):
        # Random Forest Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data)
        random_forest = RandomForestAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date,
                                              (50, 'sqrt', 5, 2, 1, True, "squared_error", None))
        return self.run_model(random_forest, "Random Forest Regression")

    def run_bayesian(self, ticker):
        # Bayesian Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data)
        bayesian = BayesianRegressionAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date,
                                               (100, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, True, True, True))
        return self.run_model(bayesian, "Bayesian Ridge Regression")

    def run_monte_carlo(self, ticker, num_of_simulations):
        # Monte Carlo Simulation:
        data = self.getDataForTicker(ticker, self.data)
        # Create a list of dates that includes weekdays only:
        weekdays = self.getWeekDays()
        monte = MonteCarloSimulation(num_of_simulations, self.prediction_date, data["Adj Close"], weekdays,
                                     self.hold_duration, self.start_date)
        # print("Monte Carlo Simulation Evaluation:")
        # mse, rmse, mae, mape, r2 = monte.evaluateModel()
        # monte.printEvaluation(mse, rmse, mae, mape, r2)
        results, s_0 = monte.makeMCPrediction(monte.get_data_for_prediction())
        # plot_labels = monte.plotSimulation(results, s_0)
        # monte.printProbabilities(plot_labels, results, s_0)
        result = monte.displayResults(results, s_0)
        return result

    def run_arima(self, ticker):
        # ARIMA:
        data = self.getDataForTicker(ticker, self.data)
        aapl = yf.Ticker(ticker)
        today_data = aapl.history(period="1d")
        arima = ARIMAAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date, today_data, [20])
        # print("ARIMA Evaluation:")
        # mse, rmse, mae, mape, r2 = arima.evaluateModel()
        # arima.printEvaluation(mse, rmse, mae, mape, r2)
        data_for_prediction = arima.get_data_for_prediction()
        predictions = arima.predict_price(data_for_prediction)
        # arima.plot_arima(predictions, data_for_prediction)
        return predictions

    def run_lstm(self, ticker):
        # LSTM:
        data = self.getDataForTicker(ticker, self.data)
        lstm = LSTMAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date, (3, 50, 0.2, 25, 'adam',
                                                                                               'mean_squared_error'))
        # mse, rmse, mae, mape, r2 = lstm.evaluateModel()
        # print("LSTM Evaluation:")
        # lstm.printEvaluation(mse, rmse, mae, mape, r2)
        prediction = lstm.predict_price(lstm.get_data_for_prediction())
        return prediction

    def get_esg_scores(self):
        # ESG Scores:
        esg = ESGScores(self.tickers)

    def plot_moving_average_graph(self, ticker):
        data = self.getDataForTicker(ticker, self.data)
        self.plotMovingAverage(data, ticker)

    def tune_hyperparameters(self, data, num_of_simulations):
        weekdays = self.getWeekDays()
        parameter_tester = ParameterTester(self.hold_duration, data, self.prediction_date, self.start_date,
                                           num_of_simulations, weekdays)

    def calculate_risk_metrics(self, ticker):
        # data = self.getDataForTicker(ticker, self.data)
        # Risk metrics:
        risk_metrics = RiskMetrics(self.tickers, self.investments, self.data["Adj Close"])
        # Display Risk Metrics results:
        vol, cat = risk_metrics.calculateVolatility(ticker)
        print("Risk Metrics:")
        print("Volatility:", cat, "(" + str(vol) + ")")
        print(risk_metrics.calculatePortfolioVolatility())
        print(risk_metrics.calculateSharpeRatio(ticker))
        print("VaR: " + str(risk_metrics.calculateVaR(ticker, 0.95, vol)))

    def get_volatility(self, ticker):
        data = self.getDataForTicker(ticker, self.data)
        risk_metrics = RiskMetrics(self.tickers, self.investments, self.data["Adj Close"])
        return risk_metrics.calculateVolatility(ticker)

    def run_model(self, model, model_name):
        # mse, rmse, mae, mape, r2 = model.evaluateModel()
        # print(model_name, "Evaluation:")
        # model.printEvaluation(mse, rmse, mae, mape, r2)
        predicted_price = model.predict_price()
        return predicted_price  # , mse, rmse, mae, mape, r2

    def downloadData(self, start, end):
        data = yf.download(self.tickers, start=start, end=end)
        return data

    def getDataForTicker(self, ticker, data):
        if len(self.tickers) >= 2:
            ticker_data = pd.DataFrame()
            for col, ti in data.columns:
                ticker_data[col] = data[col][ticker]
            return ticker_data
        return data

    # TODO: move to MonteCarlo class:
    def getWeekDays(self):
        if self.hold_duration == "1d":
            weekdays = pd.to_datetime([self.prediction_date])
        elif self.hold_duration == "1w":
            weekdays = pd.date_range(self.prediction_date - relativedelta(days=6), self.prediction_date,
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        else:
            weekdays = pd.date_range(self.prediction_date - relativedelta(months=1) + relativedelta(days=1),
                                     self.prediction_date,
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        return len(weekdays)

    # TODO: moving_avg_value set by user
    def plotMovingAverage(self, data, ticker):
        data["MA"] = data["Adj Close"].rolling(window=self.moving_avg_value).mean()

        plt.plot(data.index, data["MA"], color='green', label=f'{self.moving_avg_value}-Day Moving Average')
        plt.title(f'Moving Average of {ticker} stock')
        plt.xlabel('Date')
        plt.ylabel('Moving Average of Adjusted Close Price')
        plt.xticks(fontsize=9, rotation=340)
        plt.legend()
        plt.show()


# calc = Controller("1m")
