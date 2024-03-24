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

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        today = date.today()
        self.end_date = today

        # TODO: choose historical date range here using user's preferred investment behaviour:

        start_date_1d = today - relativedelta(months=18)
        prediction_date_1d = today + relativedelta(days=1)
        while prediction_date_1d.weekday() >= 5:
            prediction_date_1d = prediction_date_1d + relativedelta(days=1)

        start_date_1w = today - relativedelta(years=2)
        prediction_date_1w = today + relativedelta(days=7)

        start_date_1m = today - relativedelta(years=4)
        prediction_date_1m = today + relativedelta(months=1)

        self.start_dates = {
            "1d": start_date_1d,
            "1w": start_date_1w,
            "1m": start_date_1m
        }

        self.prediction_dates = {
            "1d": prediction_date_1d,
            "1w": prediction_date_1w,
            "1m": prediction_date_1m
        }

        self.moving_avg_values = {
            "1d": 5,
            "1w": 20,
            "1m": 80
        }

        self.data = {
            "1d": pd.DataFrame(),
            "1w": pd.DataFrame(),
            "1m": pd.DataFrame()
        }
        self.today = today

        self.tickers_and_investments = {}
        self.tickers_and_long_or_short = {}

        self.linear_regression_results = {"1d": {}, "1w": {}, "1m": {}}
        self.random_forest_results = {"1d": {}, "1w": {}, "1m": {}}
        self.bayesian_results = {"1d": {}, "1w": {}, "1m": {}}
        self.monte_carlo_results = {"1d": {}, "1w": {}, "1m": {}}
        self.lstm_results = {"1d": {}, "1w": {}, "1m": {}}
        self.arima_results = {"1d": {}, "1w": {}, "1m": {}}
        self.volatilities = {"1d": {}, "1w": {}, "1m": {}}

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
        # print(self.algorithms_with_indices)

        self.portfolio_results = {}

    def add_ticker(self, ticker, investment, is_long):
        ticker = ticker.upper()
        if ticker not in self.tickers_and_investments.keys():
            self.tickers_and_investments[ticker] = investment
            self.tickers_and_long_or_short[ticker] = is_long
            # Retrieve historical data from Yahoo! Finance:
            for hold_dur in self.start_dates.keys():
                self.data[hold_dur] = self.downloadData(self.start_dates[hold_dur], self.end_date)

    def remove_ticker(self, ticker):
        del self.tickers_and_investments[ticker]
        del self.tickers_and_long_or_short[ticker]
        if len(self.tickers_and_investments) > 0:
            for hold_dur in self.start_dates.keys():
                # self.data[hold_dur].drop(ticker, axis=1, inplace=True)
                self.data[hold_dur].drop(columns=ticker, level=1, inplace=True)
        else:
            self.data = {
                "1d": pd.DataFrame(),
                "1w": pd.DataFrame(),
                "1m": pd.DataFrame()
            }

    def run_algorithm(self, ticker, algorithm_index, hold_duration):
        print("called")
        if algorithm_index == 0:
            return self.run_linear_regression(ticker, hold_duration)
        elif algorithm_index == 1:
            return self.run_random_forest(ticker, hold_duration)
        elif algorithm_index == 2:
            return self.run_bayesian(ticker, hold_duration)
        elif algorithm_index == 3:
            return self.run_monte_carlo(ticker, hold_duration)
        elif algorithm_index == 4:
            return self.run_lstm(ticker, hold_duration)
        elif algorithm_index == 5:
            return self.run_arima(ticker, hold_duration)

    def run_linear_regression(self, ticker, hold_duration):
        # Linear Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        linear_regression = LinearRegressionAlgorithm(hold_duration, data, self.prediction_dates[hold_duration],
                                                      self.start_dates[hold_duration], (False,),
                                                      self.tickers_and_long_or_short[ticker],
                                                      self.tickers_and_investments[ticker])
        prediction = self.run_model(linear_regression, "Linear Regression")
        self.linear_regression_results[hold_duration][ticker] = prediction
        return prediction

    def run_random_forest(self, ticker, hold_duration):
        # Random Forest Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        if hold_duration == "1d":
            params = (50, 0.5, 10, 2, 1, True, "squared_error", 42)
        elif hold_duration == "1w":
            params = (100, 0.5, 10, 2, 2, True, "squared_error", 42)
        else:
            params = (100, 0.5, 20, 2, 2, True, "squared_error", 42)
        random_forest = RandomForestAlgorithm(hold_duration, data, self.prediction_dates[hold_duration],
                                              self.start_dates[hold_duration], params,
                                              self.tickers_and_long_or_short[ticker], self.tickers_and_investments[ticker])
        prediction = self.run_model(random_forest, "Random Forest Regression")
        self.random_forest_results[hold_duration][ticker] = prediction
        return prediction

    def run_bayesian(self, ticker, hold_duration):
        # Bayesian Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        if hold_duration == "1d":
            tol = 0.001
        else:
            tol = 1e-5
        bayesian = BayesianRegressionAlgorithm(hold_duration, data, self.prediction_dates[hold_duration],
                                               self.start_dates[hold_duration], (100, tol, 0.0001, 1e-6, 1e-6,
                                                                                 1e-4, True, False, True),
                                               self.tickers_and_long_or_short[ticker], self.tickers_and_investments[ticker])
        prediction = self.run_model(bayesian, "Bayesian Ridge Regression")
        self.bayesian_results[hold_duration][ticker] = prediction
        return prediction

    def run_monte_carlo(self, ticker, hold_duration):
        # Monte Carlo Simulation:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        # Create a list of dates that includes weekdays only:
        weekdays = self.getWeekDays(hold_duration)
        monte = MonteCarloSimulation(10000, self.prediction_dates[hold_duration], data["Adj Close"],
                                     weekdays, hold_duration, self.start_dates[hold_duration],
                                     self.tickers_and_long_or_short[ticker])
        # print("Monte Carlo Simulation Evaluation:")
        # mse, rmse, mae, mape, r2 = monte.evaluateModel()
        # monte.printEvaluation(mse, rmse, mae, mape, r2)
        results, s_0 = monte.makeMCPrediction(monte.get_data_for_prediction())
        # plot_labels = monte.plotSimulation(results, s_0)
        # monte.printProbabilities(plot_labels, results, s_0)
        result = monte.displayResults(results, s_0)

        self.monte_carlo_results[hold_duration][ticker] = result
        return result

    def run_arima(self, ticker, hold_duration):
        # ARIMA:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        aapl = yf.Ticker(ticker)
        today_data = aapl.history(period="1d")
        if hold_duration == "1d":
            params = (20, 2, 1, 1)
        elif hold_duration == "1w":
            params = (20, 2, 1, 2)
        else:
            params = (20, 0, 1, 0)
        arima = ARIMAAlgorithm(hold_duration, data, self.prediction_dates[hold_duration], self.start_dates[hold_duration],
                               today_data, params, self.tickers_and_long_or_short[ticker],
                               self.tickers_and_investments[ticker])
        # print("ARIMA Evaluation:")
        # mse, rmse, mae, mape, r2 = arima.evaluateModel()
        # arima.printEvaluation(mse, rmse, mae, mape, r2)
        data_for_prediction = arima.get_data_for_prediction()
        predictions = arima.predict_price(data_for_prediction)
        # arima.plot_arima(predictions, data_for_prediction)

        self.arima_results[hold_duration][ticker] = predictions
        return predictions

    def run_lstm(self, ticker, hold_duration):
        # LSTM:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        if hold_duration == "1d":
            params = (2, 50, 0, 25, 'adam', 'mean_squared_error', 50)
        elif hold_duration == "1w":
            params = (2, 50, 0.2, 25, 'adam', 'mean_squared_error', 50)
        else:
            params = (3, 50, 0, 25, 'adam', 'mean_squared_error', 50)
        lstm = LSTMAlgorithm(hold_duration, data, self.prediction_dates[hold_duration], self.start_dates[hold_duration],
                             params, self.tickers_and_long_or_short[ticker], self.tickers_and_investments[ticker])
        # mse, rmse, mae, mape, r2 = lstm.evaluateModel()
        # print("LSTM Evaluation:")
        # lstm.printEvaluation(mse, rmse, mae, mape, r2)
        prediction = lstm.predict_price(lstm.get_data_for_prediction())[0][0]
        self.lstm_results[hold_duration][ticker] = prediction
        return prediction

    def get_esg_scores(self):
        # ESG Scores:
        esg = ESGScores(self.tickers_and_investments.keys())

    def plot_moving_average_graph(self, ticker, hold_duration):
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        self.plotMovingAverage(data, ticker, hold_duration)

    def tune_hyperparameters(self, ticker, num_of_simulations, hold_duration):
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        weekdays = self.getWeekDays(hold_duration)
        aapl = yf.Ticker(ticker)
        today_data = aapl.history(period="1d")
        parameter_tester = ParameterTester(hold_duration, data, self.prediction_dates[hold_duration],
                                           self.start_dates[hold_duration], today_data,
                                           num_of_simulations, weekdays)

    def calculate_risk_metrics(self, ticker, hold_duration):
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        data = self.data[hold_duration]["Adj Close"]
        # Risk metrics:
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        # Display Risk Metrics results:
        vol, cat = risk_metrics.calculateVolatility(ticker)
        portfolio_vol, portfolio_vol_cat = risk_metrics.calculatePortfolioVolatility()
        portfolio_sharpe, portfolio_sharpe_cat = risk_metrics.calculatePortfolioSharpeRatio(portfolio_vol)
        portfolio_VaR = risk_metrics.calculatePortfolioVaR(0.95, portfolio_vol, hold_duration)
        sharpe_ratio = risk_metrics.calculateSharpeRatio(ticker)
        VaR = risk_metrics.calculateVaR(ticker, 0.95, vol)
        print("Risk Metrics:")
        print("Volatility:", cat, "(" + str(vol) + ")")
        print("Portfolio Volatility:", portfolio_vol, portfolio_vol_cat)
        print("Portfolio Sharpe Ratio:", portfolio_sharpe, portfolio_sharpe_cat)
        print("Portfolio VaR:", portfolio_VaR)
        print(sharpe_ratio)
        print("VaR: " + str(VaR))

        self.volatilities[hold_duration][ticker] = vol
        self.VaRs[ticker] = VaR
        self.sharpe_ratios[ticker] = sharpe_ratio

    def get_volatility(self, ticker, hold_duration):
        data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(), data)
        volatility = risk_metrics.calculateVolatility(ticker)
        self.volatilities[hold_duration][ticker] = volatility
        return volatility

    def get_portfolio_volatility(self, hold_duration):
        data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
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

    def calculate_portfolio_result(self, index, hold_duration):
        algorithm_name = self.algorithms_with_indices[index]
        algorithm_results = self.results[algorithm_name][hold_duration]
        final_result = 0
        for ticker in algorithm_results.keys():
            final_result += algorithm_results[ticker]
        self.portfolio_results[index] = final_result
        return final_result

    def calculate_portfolio_monte_carlo(self, hold_duration):
        total_positives = 0
        total_negatives = 0
        total_negatives2 = 0
        total_investments = 0
        for ticker in self.monte_carlo_results[hold_duration].keys():
            monte_result = self.monte_carlo_results[hold_duration][ticker]
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
        ticker_data = pd.DataFrame()
        if len(self.tickers_and_investments.keys()) > 1:
            for col in data.columns.levels[0]:
                ticker_data[col] = data[col][ticker]
            return ticker_data
        return data

    # TODO: move to MonteCarlo class:
    def getWeekDays(self, hold_duration):
        if hold_duration == "1d":
            weekdays = pd.to_datetime([self.prediction_dates[hold_duration]])
        elif hold_duration == "1w":
            weekdays = pd.date_range(self.prediction_dates[hold_duration] - relativedelta(days=6), self.prediction_dates[hold_duration],
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        else:
            weekdays = pd.date_range(self.prediction_dates[hold_duration] - relativedelta(months=1) + relativedelta(days=1),
                                     self.prediction_dates[hold_duration],
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        return len(weekdays)

    # TODO: moving_avg_value set by user
    def plotMovingAverage(self, data, ticker, hold_duration):
        data["MA"] = data["Adj Close"].rolling(window=self.moving_avg_values[hold_duration]).mean()

        plt.plot(data.index, data["MA"], color='green',
                 label=f'{self.moving_avg_values[hold_duration]}-Day Moving Average')
        plt.title(f'Moving Average of {ticker} stock')
        plt.xlabel('Date')
        plt.ylabel('Moving Average of Adjusted Close Price')
        plt.xticks(fontsize=9, rotation=340)
        plt.legend()
        plt.show()


# calc = Controller("1m")
