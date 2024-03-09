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

    algorithms = ["linear_regression", "random_forest", "bayesian", "monte_carlo", "lstm", "arima"]

    def __init__(self, hold_duration):
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

        self.data = pd.DataFrame()
        self.today = today

        self.tickers_and_investments = {}
        self.tickers_and_long_or_short = {}

        self.linear_regression_results = {}
        self.random_forest_results = {}
        self.bayesian_results = {}
        self.monte_carlo_results = {}
        self.lstm_results = {}
        self.arima_results = {}
        self.volatilities = {}

        # TODO: same processing for VaRs and Sharpe Ratios
        self.VaRs = {}
        self.sharpe_ratios = {}

        self.results = {
            "linear_regression": self.linear_regression_results,
            "random_forest": self.random_forest_results,
            "bayesian": self.bayesian_results,
            "monte_carlo": self.monte_carlo_results,
            "lstm": self.lstm_results,
            "arima": self.arima_results,
            "volatility": self.volatilities,
            "VaR": self.VaRs,
            "sharpe_ratio": self.sharpe_ratios
        }

        self.algorithms_with_indices = {}
        for index, name in enumerate(self.results.keys()):
            self.algorithms_with_indices[index] = name
        print(self.algorithms_with_indices)

        self.portfolio_results = {}

    def add_ticker(self, ticker, investment, is_long):
        ticker = ticker.upper()
        if ticker not in self.tickers_and_investments.keys():
            self.tickers_and_investments[ticker] = investment
            self.tickers_and_long_or_short[ticker] = is_long
            # Retrieve historical data from Yahoo! Finance:
            self.data = self.downloadData(self.start_date, self.end_date)

    def remove_ticker(self, ticker):
        del self.tickers_and_investments[ticker]
        del self.tickers_and_long_or_short[ticker]
        if len(self.tickers_and_investments) > 0:
            self.data.drop(ticker, axis=1, inplace=True)
        else:
            self.data = pd.DataFrame()

    def run_algorithm(self, ticker, algorithm_index):
        print("called")
        if algorithm_index == 0:
            return self.run_linear_regression(ticker)
        elif algorithm_index == 1:
            return self.run_random_forest(ticker)
        elif algorithm_index == 2:
            return self.run_bayesian(ticker)
        elif algorithm_index == 3:
            return self.run_monte_carlo(ticker)
        elif algorithm_index == 4:
            return self.run_lstm(ticker)
        elif algorithm_index == 5:
            return self.run_arima(ticker)

    def run_linear_regression(self, ticker):
        # Linear Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data)
        linear_regression = LinearRegressionAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date,
                                                      (False,), self.tickers_and_long_or_short[ticker],
                                                      self.tickers_and_investments[ticker])
        prediction = self.run_model(linear_regression, "Linear Regression")
        self.linear_regression_results[ticker] = prediction
        return prediction

    def run_random_forest(self, ticker):
        # Random Forest Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data)
        random_forest = RandomForestAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date,
                                              (50, 'sqrt', 5, 2, 1, True, "squared_error", None),
                                              self.tickers_and_long_or_short[ticker], self.tickers_and_investments[ticker])
        prediction = self.run_model(random_forest, "Random Forest Regression")
        self.random_forest_results[ticker] = prediction
        return prediction

    def run_bayesian(self, ticker):
        # Bayesian Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data)
        bayesian = BayesianRegressionAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date,
                                               (100, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, True, True, True),
                                               self.tickers_and_long_or_short[ticker], self.tickers_and_investments[ticker])
        prediction = self.run_model(bayesian, "Bayesian Ridge Regression")
        self.bayesian_results[ticker] = prediction
        return prediction

    def run_monte_carlo(self, ticker):
        # Monte Carlo Simulation:
        data = self.getDataForTicker(ticker, self.data)
        # Create a list of dates that includes weekdays only:
        weekdays = self.getWeekDays()
        monte = MonteCarloSimulation(10000, self.prediction_date, data["Adj Close"], weekdays,
                                     self.hold_duration, self.start_date, self.tickers_and_long_or_short[ticker])
        # print("Monte Carlo Simulation Evaluation:")
        # mse, rmse, mae, mape, r2 = monte.evaluateModel()
        # monte.printEvaluation(mse, rmse, mae, mape, r2)
        results, s_0 = monte.makeMCPrediction(monte.get_data_for_prediction())
        # plot_labels = monte.plotSimulation(results, s_0)
        # monte.printProbabilities(plot_labels, results, s_0)
        result = monte.displayResults(results, s_0)

        self.monte_carlo_results[ticker] = result
        return result

    def run_arima(self, ticker):
        # ARIMA:
        data = self.getDataForTicker(ticker, self.data)
        aapl = yf.Ticker(ticker)
        today_data = aapl.history(period="1d")
        arima = ARIMAAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date, today_data,
                               [20, 1, 1, 1], self.tickers_and_long_or_short[ticker],
                               self.tickers_and_investments[ticker])
        # print("ARIMA Evaluation:")
        # mse, rmse, mae, mape, r2 = arima.evaluateModel()
        # arima.printEvaluation(mse, rmse, mae, mape, r2)
        data_for_prediction = arima.get_data_for_prediction()
        predictions = arima.predict_price(data_for_prediction)
        # arima.plot_arima(predictions, data_for_prediction)

        self.arima_results[ticker] = predictions
        return predictions

    def run_lstm(self, ticker):
        # LSTM:
        data = self.getDataForTicker(ticker, self.data)
        lstm = LSTMAlgorithm(self.hold_duration, data, self.prediction_date, self.start_date, (3, 50, 0.2, 25, 'adam',
                                                                                               'mean_squared_error', 10),
                             self.tickers_and_long_or_short[ticker], self.tickers_and_investments[ticker])
        # mse, rmse, mae, mape, r2 = lstm.evaluateModel()
        # print("LSTM Evaluation:")
        # lstm.printEvaluation(mse, rmse, mae, mape, r2)
        prediction = lstm.predict_price(lstm.get_data_for_prediction())[0][0]
        self.lstm_results[ticker] = prediction
        return prediction

    def get_esg_scores(self):
        # ESG Scores:
        esg = ESGScores(self.tickers_and_investments.keys())

    def plot_moving_average_graph(self, ticker):
        data = self.getDataForTicker(ticker, self.data)
        self.plotMovingAverage(data, ticker)

    def tune_hyperparameters(self, ticker, num_of_simulations):
        data = self.getDataForTicker(ticker, self.data)
        weekdays = self.getWeekDays()
        aapl = yf.Ticker(ticker)
        today_data = aapl.history(period="1d")
        parameter_tester = ParameterTester(self.hold_duration, data, self.prediction_date, self.start_date, today_data,
                                           num_of_simulations, weekdays)

    def calculate_risk_metrics(self, ticker):
        data = self.data[(self.today - relativedelta(months=6)):]["Adj Close"]
        # Risk metrics:
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(), data)
        # Display Risk Metrics results:
        vol, cat = risk_metrics.calculateVolatility(ticker)
        portfolio_vol = risk_metrics.calculatePortfolioVolatility()
        sharpe_ratio = risk_metrics.calculateSharpeRatio(ticker)
        VaR = risk_metrics.calculateVaR(ticker, 0.95, vol)
        print("Risk Metrics:")
        print("Volatility:", cat, "(" + str(vol) + ")")
        print(portfolio_vol)
        print(sharpe_ratio)
        print("VaR: " + str(VaR))

        self.volatilities[ticker] = vol
        self.VaRs[ticker] = VaR
        self.sharpe_ratios[ticker] = sharpe_ratio

    def get_volatility(self, ticker):
        data = self.data[(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(), data)
        volatility = risk_metrics.calculateVolatility(ticker)
        self.volatilities[ticker] = volatility
        return volatility

    def get_portfolio_volatility(self):
        data = self.data[(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(), data)
        portfolio_vol = risk_metrics.calculatePortfolioVolatility()
        return portfolio_vol

    def run_model(self, model, model_name):
        # mse, rmse, mae, mape, r2 = model.evaluateModel()
        # print(model_name, "Evaluation:")
        # model.printEvaluation(mse, rmse, mae, mape, r2)
        predicted_price = model.predict_price()
        print(predicted_price)
        return predicted_price[0][0]  # , mse, rmse, mae, mape, r2

    def calculate_portfolio_result(self, index):
        algorithm_name = self.algorithms_with_indices[index]
        algorithm_results = self.results[algorithm_name]
        final_result = 0
        for ticker in algorithm_results.keys():
            final_result += algorithm_results[ticker]
        self.portfolio_results[index] = final_result
        return final_result

    def calculate_portfolio_results(self):
        for algorithm in self.algorithms:
            self.calculate_portfolio_result(algorithm)
        return self.portfolio_results

    def calculate_portfolio_monte_carlo(self):
        total_positives = 0
        total_negatives = 0
        total_negatives2 = 0
        total_investments = 0
        for ticker in self.monte_carlo_results.keys():
            monte_result = self.monte_carlo_results[ticker]
            percentage = float(monte_result.split('%')[0])
            if monte_result.split()[-1] == "growth":
                total_positives += percentage * self.tickers_and_investments[ticker]
            else:
                total_negatives -= percentage * self.tickers_and_investments[ticker]
                total_negatives2 += percentage * self.tickers_and_investments[ticker]
            total_investments += self.tickers_and_investments[ticker]

        result = f"{(total_positives - total_negatives) / total_investments:.2f}"
        result2 = (total_positives - total_negatives2) / total_investments

        if result2 > 0:
            result += "% chance of growth"
        else:
            result += "% chance of fall"
        return result

    def downloadData(self, start, end):
        data = yf.download(list(self.tickers_and_investments.keys()), start=start, end=end)
        return data

    def getDataForTicker(self, ticker, data):
        if len(self.tickers_and_investments.keys()) >= 2:
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
