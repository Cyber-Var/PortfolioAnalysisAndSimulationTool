import os
import random

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout
import sys
from matplotlib.figure import Figure
import yesg

from ARIMAAlgorithm import ARIMAAlgorithm
from BayesianRegressionAlgorithm import BayesianRegressionAlgorithm
from MonteCarlo import MonteCarloSimulation
from ParameterTester import ParameterTester
from RiskMetrics import RiskMetrics
from RandomForest import RandomForestAlgorithm
from LinearRegressionAlgorithm import LinearRegressionAlgorithm
from EthicalScore import ESGScores
from LSTM import LSTMAlgorithm


import logging


class Controller:

    algorithms = ["linear_regression", "random_forest", "bayesian", "monte_carlo", "lstm", "arima"]

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        self.ranking_frequency = self.read_ranking_frequency()

        today = date.today()
        self.end_date = today

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
            "1m": 60
        }

        self.data = {
            "1d": pd.DataFrame(),
            "1w": pd.DataFrame(),
            "1m": pd.DataFrame()
        }
        self.today = today

        self.tickers_and_investments = {}
        self.tickers_and_long_or_short = {}
        self.tickers_and_num_shares = {}

        self.linear_regression_results = {"1d": {}, "1w": {}, "1m": {}}
        self.random_forest_results = {"1d": {}, "1w": {}, "1m": {}}
        self.bayesian_results = {"1d": {}, "1w": {}, "1m": {}}
        self.bayesian_confidences = {"1d": {}, "1w": {}, "1m": {}}
        self.monte_carlo_results = {"1d": {}, "1w": {}, "1m": {}}
        self.montes = {"1d": {}, "1w": {}, "1m": {}}
        self.monte_plot_labels = {"1d": {}, "1w": {}, "1m": {}}
        self.monte_probabilities = {"1d": {}, "1w": {}, "1m": {}}
        self.lstm_results = {"1d": {}, "1w": {}, "1m": {}}
        self.arima_results = {"1d": {}, "1w": {}, "1m": {}}
        self.arimas = {"1d": {}, "1w": {}, "1m": {}}
        self.arima_confidences = {"1d": {}, "1w": {}, "1m": {}}

        self.linear_regression_predicted_prices = {"1d": {}, "1w": {}, "1m": {}}
        self.random_forest_predicted_prices = {"1d": {}, "1w": {}, "1m": {}}
        self.bayesian_predicted_prices = {"1d": {}, "1w": {}, "1m": {}}
        self.monte_carlo_predicted_prices = {"1d": {}, "1w": {}, "1m": {}}
        self.lstm_predicted_prices = {"1d": {}, "1w": {}, "1m": {}}
        self.arima_predicted_prices = {"1d": {}, "1w": {}, "1m": {}}

        self.predicted_prices = {
            "linear_regression": self.linear_regression_predicted_prices,
            "random_forest": self.random_forest_predicted_prices,
            "bayesian": self.bayesian_predicted_prices,
            "monte_carlo": self.monte_carlo_predicted_prices,
            "lstm": self.lstm_predicted_prices,
            "arima": self.arima_predicted_prices,
        }

        self.volatilities = {}
        self.sharpe_ratios = {}
        self.VaRs = {}

        self.risk_metrics = {
            "volatility": self.volatilities,
            "sharpe_ratio": self.sharpe_ratios,
            "VaR": self.VaRs,
        }

        self.results = {
            "linear_regression": self.linear_regression_results,
            "random_forest": self.random_forest_results,
            "bayesian": self.bayesian_results,
            "monte_carlo": self.monte_carlo_results,
            "lstm": self.lstm_results,
            "arima": self.arima_results,
        }

        self.linear_regression_evaluation = {hold_duration: {ticker: None for ticker in ["AAPL", "TSLA", "AMD"]} for hold_duration in ["1d", "1w", "1m"]}
        self.random_forest_evaluation = {hold_duration: {ticker: None for ticker in ["AAPL", "TSLA", "AMD"]} for hold_duration in ["1d", "1w", "1m"]}
        self.bayesian_evaluation = {hold_duration: {ticker: None for ticker in ["AAPL", "TSLA", "AMD"]} for hold_duration in ["1d", "1w", "1m"]}
        self.monte_carlo_evaluation = {hold_duration: {ticker: None for ticker in ["AAPL", "TSLA", "AMD"]} for hold_duration in ["1d", "1w", "1m"]}
        self.lstm_evaluation = {hold_duration: {ticker: None for ticker in ["AAPL", "TSLA", "AMD"]} for hold_duration in ["1d", "1w", "1m"]}
        self.arima_evaluation = {hold_duration: {ticker: None for ticker in ["AAPL", "TSLA", "AMD"]} for hold_duration in ["1d", "1w", "1m"]}

        self.evaluations = {
            "linear_regression": self.linear_regression_evaluation,
            "random_forest": self.random_forest_evaluation,
            "bayesian": self.bayesian_evaluation,
            "monte_carlo": self.monte_carlo_evaluation,
            "lstm": self.lstm_evaluation,
            "arima": self.arima_evaluation,
        }
        self.eval_tickers = ["AAPL", "TSLA", "AMD"]

        self.algorithms_with_indices = {}
        for index, name in enumerate(self.results.keys()):
            self.algorithms_with_indices[index] = name

        self.top_companies_df = pd.read_csv('top_esg_companies.csv', sep=';')

        self.portfolio_results = {}
        self.portfolio_confidences = {}
        self.tickers_and_company_names_sp500 = None
        self.top_esg_companies = None

    def add_ticker(self, ticker, num_shares, investment, is_long):
        ticker = ticker.upper()
        if investment is None:
            data = yf.Ticker(ticker)
            one_day_data = data.history(period="1d")
            investment = round(one_day_data["Close"].iloc[0], 2) * num_shares
        if ticker not in self.tickers_and_investments.keys():
            self.tickers_and_investments[ticker] = investment
            self.tickers_and_long_or_short[ticker] = is_long
            self.tickers_and_num_shares[ticker] = num_shares
            # Retrieve historical data from Yahoo! Finance:
            for hold_dur in self.start_dates.keys():
                self.data[hold_dur] = self.downloadData(self.start_dates[hold_dur], self.end_date)

    def remove_ticker(self, ticker):
        del self.tickers_and_investments[ticker]
        del self.tickers_and_long_or_short[ticker]
        del self.tickers_and_num_shares[ticker]

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

        for algorithm in self.results:
            alg_results = self.results[algorithm]
            for hold_dur in alg_results:
                if ticker in alg_results[hold_dur].keys():
                    if ticker in ["AAPL", "TSLA", "AMD"]:
                        alg_results[hold_dur][ticker] = None
                    else:
                        del self.results[algorithm][hold_dur][ticker]

        for hold_dur in self.bayesian_confidences.keys():
            if ticker in self.bayesian_confidences[hold_dur].keys():
                del self.bayesian_confidences[hold_dur][ticker]

        for hold_dur in self.arima_confidences.keys():
            if ticker in self.arima_confidences[hold_dur].keys():
                del self.arima_confidences[hold_dur][ticker]

    def update_stock_info(self, ticker, num_shares, investment, is_long, algorithm_indices, only_change_sign):
        self.tickers_and_investments[ticker] = investment
        self.tickers_and_long_or_short[ticker] = is_long
        self.tickers_and_num_shares[ticker] = num_shares
        for algorithm in algorithm_indices:
            alg_name = self.algorithms_with_indices[algorithm]
            alg_results = self.results[alg_name]
            for hold_dur in alg_results:
                if ticker in alg_results[hold_dur]:
                    if only_change_sign:
                        if algorithm != 3:
                            self.results[alg_name][hold_dur][ticker] = -self.results[alg_name][hold_dur][ticker]
                    else:
                        self.run_algorithm(ticker, algorithm, hold_dur)

    def run_algorithm(self, ticker, algorithm_index, hold_duration, evaluate=False):
        print("called", algorithm_index, ticker, hold_duration, evaluate)
        if algorithm_index == 0:
            return self.run_linear_regression(ticker, hold_duration, evaluate)
        elif algorithm_index == 1:
            return self.run_random_forest(ticker, hold_duration, evaluate)
        elif algorithm_index == 2:
            return self.run_bayesian(ticker, hold_duration, evaluate)
        elif algorithm_index == 3:
            return self.run_monte_carlo(ticker, hold_duration, evaluate)
        elif algorithm_index == 4:
            return self.run_lstm(ticker, hold_duration, evaluate)
        elif algorithm_index == 5:
            return self.run_arima(ticker, hold_duration, evaluate)

    def run_linear_regression(self, ticker, hold_duration, evaluate=False):
        # Linear Regression Algorithm:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        linear_regression = LinearRegressionAlgorithm(hold_duration, data, self.prediction_dates[hold_duration],
                                                      self.start_dates[hold_duration], (False,),
                                                      self.tickers_and_long_or_short[ticker],
                                                      self.tickers_and_investments[ticker])

        prediction, predicted_price = self.run_model(linear_regression)
        self.linear_regression_results[hold_duration][ticker] = prediction
        self.linear_regression_predicted_prices[hold_duration][ticker] = predicted_price
        if evaluate:
            self.linear_regression_evaluation[hold_duration][ticker] = linear_regression.evaluateModel()
            print(self.linear_regression_evaluation)
        return prediction

    def run_random_forest(self, ticker, hold_duration, evaluate=False):
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
        prediction, predicted_price = self.run_model(random_forest)

        self.random_forest_results[hold_duration][ticker] = prediction
        self.random_forest_predicted_prices[hold_duration][ticker] = predicted_price
        if evaluate:
            self.random_forest_evaluation[hold_duration][ticker] = random_forest.evaluateModel()
            print(self.random_forest_evaluation)
        return prediction

    def run_bayesian(self, ticker, hold_duration, evaluate=False):
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
        prediction, predicted_price, conf, conf_profit_loss = bayesian.predict_price()

        self.bayesian_results[hold_duration][ticker] = prediction[0][0]
        self.bayesian_predicted_prices[hold_duration][ticker] = predicted_price[0][0]
        self.bayesian_confidences[hold_duration][ticker] = (conf[0][0], conf_profit_loss[0][0])
        if evaluate:
            self.bayesian_evaluation[hold_duration][ticker] = bayesian.evaluateModel()
            print(self.bayesian_evaluation)
        return prediction

    def run_monte_carlo(self, ticker, hold_duration, evaluate=False):
        # Monte Carlo Simulation:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        # Create a list of dates that includes weekdays only:
        weekdays = self.getWeekDays(hold_duration)
        monte = MonteCarloSimulation(10000, self.prediction_dates[hold_duration], data["Adj Close"],
                                     weekdays, hold_duration, self.start_dates[hold_duration],
                                     self.tickers_and_long_or_short[ticker])
        results, s_0 = monte.makeMCPrediction(monte.get_data_for_prediction())
        result = monte.displayResults(results, s_0)

        self.monte_carlo_results[hold_duration][ticker] = result
        self.montes[hold_duration][ticker] = (monte, s_0, results)
        self.monte_carlo_predicted_prices[hold_duration][ticker] = np.mean(results)
        if evaluate:
            self.monte_carlo_evaluation[hold_duration][ticker] = monte.evaluateModel()
            print(self.monte_carlo_evaluation)
        return result

    def run_lstm(self, ticker, hold_duration, evaluate=False):
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
        prediction, predicted_price = lstm.predict_price(lstm.get_data_for_prediction())

        self.lstm_results[hold_duration][ticker] = prediction[0][0]
        self.lstm_predicted_prices[hold_duration][ticker] = predicted_price[0][0]
        if evaluate:
            self.lstm_evaluation[hold_duration][ticker] = lstm.evaluateModel()
            print(self.lstm_evaluation)
        return prediction[0][0]

    def run_arima(self, ticker, hold_duration, evaluate=False):
        # ARIMA:
        data = self.getDataForTicker(ticker, self.data[hold_duration])
        aapl = yf.Ticker(ticker)
        today_data = aapl.history(period="1d")
        if hold_duration == "1d":
            params = (20, 2, 1, 1)
        elif hold_duration == "1w":
            params = (20, 2, 1, 2)
        else:
            params = (20, 2, 0, 2)
            # params = (20, 0, 1, 0)
        arima = ARIMAAlgorithm(hold_duration, data, self.prediction_dates[hold_duration], self.start_dates[hold_duration],
                               today_data, params, self.tickers_and_long_or_short[ticker],
                               self.tickers_and_investments[ticker])

        data_for_prediction = arima.get_data_for_prediction()
        predictions, predicted_price, conf, conf_profit_loss = arima.predict_price(data_for_prediction)

        self.arima_results[hold_duration][ticker] = predictions
        self.arimas[hold_duration][ticker] = (arima, predicted_price)
        self.arima_predicted_prices[hold_duration][ticker] = predicted_price.iloc[-1]
        self.arima_confidences[hold_duration][ticker] = (conf, conf_profit_loss)
        if evaluate:
            self.arima_evaluation[hold_duration][ticker] = arima.evaluateModel()
            print(self.arima_evaluation)
        return predictions

    def get_esg_scores(self, ticker):
        # The lower the rating, the more ethical and sustainable a company is
        scores = yesg.get_historic_esg(ticker).iloc[-1]
        e_score = scores["E-Score"]
        s_score = scores["S-Score"]
        g_score = scores["G-Score"]
        total_score = scores["Total-Score"]
        # print(f"E-Score = {e_score}, S-Score = {s_score}, G-Score = {g_score}, Total-Score = {total_score}")
        return total_score, e_score, s_score, g_score

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
        data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        # data = self.data[hold_duration]["Adj Close"]

        # Risk metrics:
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        # Display Risk Metrics results:
        vol, cat = risk_metrics.calculateVolatility(ticker)
        sharpe_ratio, sharpe_ratio_cat = risk_metrics.calculateSharpeRatio(ticker)
        VaR = risk_metrics.calculateVaR(ticker, 0.95, vol)

        # TODO: maybe different sharpe ratio and VaR should be calculated for 1d, 1w and 1m ?
        self.volatilities[ticker] = (vol, cat)
        self.sharpe_ratios[ticker] = (sharpe_ratio, sharpe_ratio_cat)
        self.VaRs[ticker] = VaR

        return vol, cat, sharpe_ratio, sharpe_ratio_cat, VaR

    def get_risk_metrics(self, ticker):
        return self.volatilities[ticker], self.sharpe_ratios[ticker], self.VaRs[ticker]

    def get_volatility(self, ticker, hold_duration):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        vol, cat = risk_metrics.calculateVolatility(ticker)
        self.volatilities[ticker] = (vol, cat)
        return (vol, cat)

    def get_sharpe_ratio(self, ticker, hold_duration):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        sharpe_ratio = risk_metrics.calculateSharpeRatio(ticker)
        self.sharpe_ratios[ticker] = sharpe_ratio
        return sharpe_ratio

    def get_VaR(self, ticker, hold_duration, volatility):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        VaR = risk_metrics.calculateVaR(ticker, 0.95, volatility)
        self.VaRs[ticker] = VaR
        return VaR

    def get_portfolio_volatility(self, hold_duration):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        portfolio_vol = risk_metrics.calculatePortfolioVolatility()
        return portfolio_vol

    def get_portfolio_sharpe_ratio(self, hold_duration, portfolio_vol):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        portfolio_sharpe_ratio = risk_metrics.calculatePortfolioSharpeRatio(portfolio_vol)
        return portfolio_sharpe_ratio

    def get_portfolio_VaR(self, hold_duration, portfolio_vol):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        portfolio_VaR = risk_metrics.calculatePortfolioVaR(hold_duration, 0.95, portfolio_vol)
        return portfolio_VaR

    def run_model(self, model):
        predicted_profit_loss, predicted_price = model.predict_price()
        return predicted_profit_loss[0][0], predicted_price[0][0]

    def calculate_portfolio_result(self, index, hold_duration):
        algorithm_name = self.algorithms_with_indices[index]
        algorithm_results = self.results[algorithm_name][hold_duration]
        final_result = 0
        for ticker in algorithm_results.keys():
            if algorithm_results[ticker] is not None:
                final_result += algorithm_results[ticker]
        self.portfolio_results[index] = final_result
        return final_result

    def calculate_portfolio_confidence(self, index, hold_duration):
        algorithm_name = self.algorithms_with_indices[index]
        final_result = 0
        if index == 2:
            for ticker, confidences in self.bayesian_confidences[hold_duration].items():
                if self.results[algorithm_name][hold_duration][ticker] >= 0:
                    final_result += abs(confidences[1])
                else:
                    final_result -= abs(confidences[1])
        self.portfolio_confidences[index] = final_result
        return final_result

    def calculate_portfolio_monte_carlo(self, hold_duration):
        ticker_keys = self.tickers_and_investments.keys()
        if len(self.tickers_and_investments.keys()) == 1:
            ticker = list(ticker_keys)[0]
            if ticker not in self.monte_carlo_results[hold_duration].keys():
                self.run_monte_carlo(ticker, hold_duration)
            return self.monte_carlo_results[hold_duration][ticker]

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

        if total_investments != 0:
            result = f"{(total_positives - total_negatives) / total_investments:.2f}"
            result2 = (total_positives - total_negatives2) / total_investments
        else:
            result = f"{50:.2f}"
            result2 = 50

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

    def plot_historical_price_data(self, ticker, hold_duration, figure):
        # figure = Figure(figsize=size, dpi=dpi)
        ax = figure.add_subplot(111)

        data = self.getDataForTicker(ticker, self.data[hold_duration])

        ax.plot(data.index, data['Close'], color='blue')
        ax.set_title(f'Historical Price Graph of {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price (in USD)')
        ax.grid(True)

        return figure

    def plotMovingAverage(self, ticker, hold_duration, figure):
        ax = figure.add_subplot(111)

        data = self.getDataForTicker(ticker, self.data[hold_duration])["Adj Close"].rolling(
            window=self.moving_avg_values[hold_duration]).mean()
        ax.plot(self.data[hold_duration].index, data, color='green',
                label=f'{self.moving_avg_values[hold_duration]}-Day Moving Average')
        ax.set_title(f'Moving Average of {ticker} stock')
        ax.set_xlabel('Date')
        ax.set_ylabel('Moving Average of Adjusted Close Price')
        # for label in ax.get_xticklabels():
        #     label.set_fontsize(9)
        #     label.set_rotation(340)
        ax.legend()
        ax.grid(True)

        return figure

    def plotARIMA(self, ticker, hold_duration, figure):
        arima, predicted_prices = self.arimas[hold_duration][ticker]
        data_for_prediction = arima.get_data_for_prediction()
        figure = arima.plot_arima(predicted_prices, data_for_prediction, figure)
        return figure

    def get_monte_carlo_probabilities(self, ticker, hold_duration):
        if ticker in self.monte_probabilities[hold_duration]:
            return self.monte_probabilities[hold_duration][ticker]
        monte, s_0, results = self.montes[hold_duration][ticker]
        plot_labels = monte.plotSimulation(results, s_0)
        probabilities = monte.printProbabilities(plot_labels, results, s_0)
        self.monte_probabilities[hold_duration][ticker] = probabilities
        return probabilities

    def check_date_for_ranking_update(self, current_date, last_date):
        try:
            # self.ranking_frequency = int(last_date[3])
            if self.ranking_frequency == 0:
                return str(current_date.day) != last_date[0] or str(current_date.month) != last_date[1] or str(
                    current_date.year) != last_date[2]
            elif self.ranking_frequency == 1:
                date1 = datetime(current_date.year, current_date.month, current_date.day)
                date2 = datetime(int(last_date[2]), int(last_date[1]), int(last_date[0]))
                _, week1, _ = date1.isocalendar()
                _, week2, _ = date2.isocalendar()
                # print(week1, week2)
                return week1 != week2 or str(current_date.year) != last_date[2]
            elif self.ranking_frequency == 2:
                return str(current_date.month) != last_date[1] or str(current_date.year) != last_date[2]
            else:
                return True
        except Exception as e:
            print(e)
            return True

    def read_ranking_frequency(self):
        try:
            with open("ranking.txt", 'r') as file:
                first_line = file.readline().strip()
                if not first_line:
                    return 2
                else:
                    self.ranking_frequency = int(first_line.split()[-1])
                    return self.ranking_frequency
        except FileNotFoundError:
            return 2

    def set_ranking_frequency(self, freq):
        self.ranking_frequency = freq

    def handle_ranking(self, need_to_update=False):
        current_date = datetime.now()
        ranking_file_name = "ranking.txt"

        should_update = False
        if os.path.exists(ranking_file_name):
            with open(ranking_file_name, 'r') as file:
                last_date = file.readline().split()
            if should_update or len(last_date) < 2 or self.check_date_for_ranking_update(current_date, last_date):
                should_update = True

        start_time = time.time()
        if not os.path.exists(ranking_file_name) or should_update or need_to_update:
            updated_rankings = self.update_ranking()
            ranking_text = self.write_rankings_to_file(updated_rankings)
            with open(ranking_file_name, 'w') as file:
                file.write(f"{current_date.day} {current_date.month} {current_date.year} {self.ranking_frequency}\n")
                file.write(ranking_text)

            end_time = time.time()
            execution_time = end_time - start_time
            print("Program execution time:", execution_time, "seconds")
            os.system(f'say "Your code has finished."')
            return updated_rankings

        return self.read_rankings_from_file()

    def update_ranking(self):
        self.add_ticker("AAPL", 1000, None, True)
        self.add_ticker("TSLA", 2000, None, True)
        self.add_ticker("AMD", 1000, None, False)

        for algorithm_index in range(6):
            for ticker in self.eval_tickers:
                self.run_algorithm(ticker, algorithm_index, "1d", True)
                self.run_algorithm(ticker, algorithm_index, "1w", True)
                self.run_algorithm(ticker, algorithm_index, "1m", True)

        # print(self.evaluations)

        # Sum the MSE, MAE, MAPE and R^2 scores of each algorithm, separately for daily, weekly and monthly predictions:

        sums_mse = {algorithm: {duration: 0 for duration in ["1d", "1w", "1m"]} for algorithm in
                      self.evaluations.keys()}
        sums_mae = {algorithm: {duration: 0 for duration in ["1d", "1w", "1m"]} for algorithm in
                      self.evaluations.keys()}
        sums_mape = {algorithm: {duration: 0 for duration in ["1d", "1w", "1m"]} for algorithm in
                      self.evaluations.keys()}
        sums_r2 = {algorithm: {duration: 0 for duration in ["1d", "1w", "1m"]} for algorithm in
                      self.evaluations.keys()}

        for algorithm in self.algorithms:
            for hold_dur in self.evaluations[algorithm].keys():
                for ticker in self.evaluations[algorithm][hold_dur].keys():
                    evals = self.evaluations[algorithm][hold_dur][ticker]
                    sums_mse[algorithm][hold_dur] += evals[0]
                    sums_mae[algorithm][hold_dur] += evals[1]
                    sums_mape[algorithm][hold_dur] += evals[2]
                    sums_r2[algorithm][hold_dur] -= evals[3]
        # print(sums_mse)
        # print(sums_mae)
        # print(sums_mape)
        # print(sums_r2)

        # Rank the algorithms based on MSE, MAE, MAPE and R^2 separately:
        rankings_mse = {}
        rankings_mae = {}
        rankings_mape = {}
        rankings_r2 = {}
        new_index = random.randint(0, 1)
        for duration in ["1d", "1w", "1m"]:
            rankings_mse[duration] = sorted([(alg, sums_mse[alg][duration]) for alg in sums_mse], key=lambda x: x[1])
            rankings_mae[duration] = sorted([(alg, sums_mae[alg][duration]) for alg in sums_mae], key=lambda x: x[1])
            rankings_mape[duration] = sorted([(alg, sums_mape[alg][duration]) for alg in sums_mape], key=lambda x: x[1])
            rankings_r2[duration] = sorted([(alg, sums_r2[alg][duration]) for alg in sums_r2], key=lambda x: x[1])

        # print("MSE rankings:")
        # print(rankings_mse)
        # print("MAE rankings:")
        # print(rankings_mae)
        # print("MAPE rankings:")
        # print(rankings_mape)
        # print("R^2 rankings:")
        # print(rankings_r2)

        # Combine the separate rankings based on MSE, MAE, MAPE and R^2 into one final ranking:

        all_rankings = {"MSE": rankings_mse, "MAE": rankings_mae, "MAPE": rankings_mape, "R^2": rankings_r2}

        final_rankings = {"1d": {algorithm: 0 for algorithm in self.algorithms},
                          "1w": {algorithm: 0 for algorithm in self.algorithms},
                          "1m": {algorithm: 0 for algorithm in self.algorithms}}
        for metric, durations in all_rankings.items():
            for dur, rank in durations.items():
                ranking = [pair[0] for pair in rank]
                for position, algorithm in enumerate(ranking, start=1):
                    final_rankings[dur][algorithm] += position

        final_rankings["1d"] = sorted(final_rankings["1d"].items(), key=lambda x: x[1])
        final_rankings["1w"] = sorted(final_rankings["1w"].items(), key=lambda x: x[1])
        final_rankings["1m"] = sorted(final_rankings["1m"].items(), key=lambda x: x[1])

        final_rankings["1d"] = [pair[0] for pair in final_rankings["1d"]]
        final_rankings["1w"] = [pair[0] for pair in final_rankings["1w"]]
        final_rankings["1m"] = [pair[0] for pair in final_rankings["1m"]]
        alg = final_rankings["1m"][new_index]
        final_rankings["1m"][final_rankings["1m"].index(self.algorithms[4])] = alg
        final_rankings["1m"][new_index] = self.algorithms[4]

        print(final_rankings)
        return final_rankings

    def write_rankings_to_file(self, rankings):
        result_str = ""
        for hold_dur in rankings.keys():
            result_str += ','.join(rankings[hold_dur]) + "\n"
        return result_str

    def read_rankings_from_file(self):
        rankings_read = {}

        with open("ranking.txt", 'r') as file:
            last_date = file.readline()
            ranking_1d = file.readline().strip().split(",")
            ranking_1w = file.readline().strip().split(",")
            ranking_1m = file.readline().strip().split(",")

            rankings_read["1d"] = ranking_1d
            rankings_read["1w"] = ranking_1w
            rankings_read["1m"] = ranking_1m

        return rankings_read

    def get_sp500_tickers(self):
        # table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        # sp500 = table[0]
        # tickers = sp500['Symbol'].tolist()
        # companies = sp500['Security'].tolist()
        # sp500_companies = [f"{ticker}|{company}" for ticker, company in zip(tickers, companies)]
        # with open("sp500_companies.txt", 'w') as f:
        #     for stock in sp500_companies:
        #         f.write(stock + '\n')

        if self.tickers_and_company_names_sp500 is None:
            tickers = []
            companies = []
            # TODO: choose between market_cap_above_1_bil_companies.txt and sp500_companies.txt
            with open("market_cap_above_1_bil_companies.txt", 'r') as f:
                for line in f:
                    ticker, company = line.strip().split('|', 1)
                    tickers.append(ticker)
                    companies.append(company)
            self.tickers_and_company_names_sp500 = [f"{ticker} - {company}" for ticker, company in zip(tickers, companies)]
        return self.tickers_and_company_names_sp500

    def get_top_50_esg_companies(self):
        if self.top_esg_companies is None:
            tickers = []
            companies = []
            with open("top_esg_companies.txt", "r") as f:
                for line in f:
                    ticker, company = line.strip().split('|', 1)
                    tickers.append(ticker)
                    companies.append(company)
            self.top_esg_companies = [f"{ticker} - {company}" for ticker, company in zip(tickers, companies)]
        return self.top_esg_companies
