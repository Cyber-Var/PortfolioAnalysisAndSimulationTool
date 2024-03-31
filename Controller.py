import os

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
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
        self.sharpe_ratios ={"1d": {}, "1w": {}, "1m": {}}
        self.VaRs = {"1d": {}, "1w": {}, "1m": {}}

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
        prediction = self.run_model(linear_regression, "Linear Regression")
        self.linear_regression_results[hold_duration][ticker] = prediction
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
        prediction = self.run_model(random_forest, "Random Forest Regression")
        self.random_forest_results[hold_duration][ticker] = prediction
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
        prediction = self.run_model(bayesian, "Bayesian Ridge Regression")
        self.bayesian_results[hold_duration][ticker] = prediction
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
        # print("Monte Carlo Simulation Evaluation:")
        # mse, mae, mape, r2 = monte.evaluateModel()
        # monte.printEvaluation(mse, mae, mape, r2)
        results, s_0 = monte.makeMCPrediction(monte.get_data_for_prediction())
        # plot_labels = monte.plotSimulation(results, s_0)
        # monte.printProbabilities(plot_labels, results, s_0)
        result = monte.displayResults(results, s_0)

        self.monte_carlo_results[hold_duration][ticker] = result
        if evaluate:
            self.monte_carlo_evaluation[hold_duration][ticker] = monte.evaluateModel()
            print(self.monte_carlo_evaluation)
        return result

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
            params = (20, 0, 1, 0)
        arima = ARIMAAlgorithm(hold_duration, data, self.prediction_dates[hold_duration], self.start_dates[hold_duration],
                               today_data, params, self.tickers_and_long_or_short[ticker],
                               self.tickers_and_investments[ticker])
        # print("ARIMA Evaluation:")
        # mse, mae, mape, r2 = arima.evaluateModel()
        # arima.printEvaluation(mse, mae, mape, r2)
        data_for_prediction = arima.get_data_for_prediction()
        predictions = arima.predict_price(data_for_prediction)
        # arima.plot_arima(predictions, data_for_prediction)

        self.arima_results[hold_duration][ticker] = predictions
        if evaluate:
            self.arima_evaluation[hold_duration][ticker] = arima.evaluateModel()
            print(self.arima_evaluation)
        return predictions

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
        # mse, mae, mape, r2 = lstm.evaluateModel()
        # print("LSTM Evaluation:")
        # lstm.printEvaluation(mse, mae, mape, r2)
        prediction = lstm.predict_price(lstm.get_data_for_prediction())[0][0]
        self.lstm_results[hold_duration][ticker] = prediction
        if evaluate:
            self.lstm_evaluation[hold_duration][ticker] = lstm.evaluateModel()
            print(self.lstm_evaluation)
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
        sharpe_ratio, sharpe_ratio_cat = risk_metrics.calculateSharpeRatio(ticker)
        VaR = risk_metrics.calculateVaR(ticker, 0.95, vol)
        print("Risk Metrics:")
        print("Volatility:", cat, "(" + str(vol) + ")")
        print("Portfolio Volatility:", portfolio_vol, portfolio_vol_cat)
        print("Portfolio Sharpe Ratio:", portfolio_sharpe, portfolio_sharpe_cat)
        print("Portfolio VaR:", portfolio_VaR)
        print("Sharpe Ratio:", sharpe_ratio, sharpe_ratio_cat)
        print("VaR: " + str(VaR))

        self.volatilities[hold_duration][ticker] = vol
        self.VaRs[ticker] = VaR
        self.sharpe_ratios[ticker] = sharpe_ratio

    def get_volatility(self, ticker, hold_duration):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        volatility = risk_metrics.calculateVolatility(ticker)
        self.volatilities[hold_duration][ticker] = volatility
        return volatility

    def get_sharpe_ratio(self, ticker, hold_duration):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        sharpe_ratio = risk_metrics.calculateSharpeRatio(ticker)
        self.sharpe_ratios[hold_duration][ticker] = sharpe_ratio
        return sharpe_ratio

    def get_VaR(self, ticker, hold_duration, volatility):
        data = self.data[hold_duration]["Adj Close"]
        # data = self.data[hold_duration][(self.today - relativedelta(months=6)):]["Adj Close"]
        risk_metrics = RiskMetrics(self.tickers_and_investments.keys(), self.tickers_and_investments.values(),
                                   self.tickers_and_long_or_short.values(), data)
        VaR = risk_metrics.calculateVaR(ticker, 0.95, volatility)
        self.VaRs[hold_duration][ticker] = VaR
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

    def run_model(self, model, model_name):
        # mse, mae, mape, r2 = model.evaluateModel()
        # print(model_name, "Evaluation:")
        # model.printEvaluation(mse, mae, mape, r2)
        predicted_price = model.predict_price()
        print(predicted_price)
        return predicted_price[0][0]  # , mse, mae, mape, r2

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

    def handle_ranking(self, need_to_update=False):
        current_date = datetime.now()
        ranking_file_name = "ranking.txt"

        should_update = False
        if os.path.exists(ranking_file_name):
            with open(ranking_file_name, 'r') as file:
                last_date = file.readline().split()
            if should_update or len(last_date) < 2 or str(current_date.month) != last_date[0] or str(current_date.year) != last_date[1]:
                should_update = True

        start_time = time.time()
        if not os.path.exists(ranking_file_name) or should_update or need_to_update:
            updated_rankings = self.update_ranking()
            ranking_text = self.write_rankings_to_file(updated_rankings)
            with open(ranking_file_name, 'w') as file:
                file.write(f"{current_date.month} {current_date.year}\n")
                file.write(ranking_text)

            end_time = time.time()
            execution_time = end_time - start_time
            print("Program execution time:", execution_time, "seconds")
            return updated_rankings

        return self.read_rankings_from_file()

    def update_ranking(self):
        self.add_ticker("AAPL", 1000, True)
        self.add_ticker("TSLA", 2000, True)
        self.add_ticker("AMD", 1000, False)

        for algorithm_index in range(6):
            for ticker in self.eval_tickers:
                self.run_algorithm(ticker, algorithm_index, "1d", True)
                self.run_algorithm(ticker, algorithm_index, "1w", True)
                self.run_algorithm(ticker, algorithm_index, "1m", True)

        print(self.evaluations)

        # Sum the MSE, MAE, MAPE and R^2 scores of each algorithm, separately for daily, weekly and monthly predictions:

        sums_mse = {algorithm: {duration: 0 for duration in ["1d", "1w", "1m"]} for algorithm in
                      self.evaluations.keys()}
        print(sums_mse)
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
        print(sums_mse)
        print(sums_mae)
        print(sums_mape)
        print(sums_r2)

        # Rank the algorithms based on MSE, MAE, MAPE and R^2 separately:
        rankings_mse = {}
        rankings_mae = {}
        rankings_mape = {}
        rankings_r2 = {}
        for duration in ["1d", "1w", "1m"]:
            # TODO: check that this is ranked correctly:
            rankings_mse[duration] = sorted([(alg, sums_mse[alg][duration]) for alg in sums_mse], key=lambda x: x[1])
            rankings_mae[duration] = sorted([(alg, sums_mae[alg][duration]) for alg in sums_mae], key=lambda x: x[1])
            rankings_mape[duration] = sorted([(alg, sums_mape[alg][duration]) for alg in sums_mape], key=lambda x: x[1])
            rankings_r2[duration] = sorted([(alg, sums_r2[alg][duration]) for alg in sums_r2], key=lambda x: x[1])

        print("MSE rankings:")
        print(rankings_mse)
        print("MAE rankings:")
        print(rankings_mae)
        print("MAPE rankings:")
        print(rankings_mape)
        print("R^2 rankings:")
        print(rankings_r2)

        # rankings_mse = {'1d': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 71.26391312326031), ('random_forest', 336.60440620615987)], '1w': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 391.2276127608421), ('random_forest', 1029.9607039232037)], '1m': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 1939.2227765348366), ('random_forest', 2022.090700119813)]}
        # rankings_mae = {'1d': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 9.901447382031126), ('random_forest', 20.37349078540057)], '1w': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 25.37079207073901), ('random_forest', 39.775791630646296)], '1m': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('random_forest', 56.33070692269658), ('linear_regression', 57.41482192776909)]}
        # rankings_mape = {'1d': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 5.611251397569742), ('random_forest', 11.247492211399885)], '1w': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 14.398992372770188), ('random_forest', 22.664536036295004)], '1m': [('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('random_forest', 32.07457392072242), ('linear_regression', 32.339696113892124)]}
        # rankings_r2 = {'1d': [('linear_regression', -2.8833572878522746), ('random_forest', -2.4412863500683883), ('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0)], '1w': [('linear_regression', -2.2314029569409657), ('random_forest', -0.9757240123007211), ('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0)], '1m': [('random_forest', -0.32377661629677823), ('bayesian', 0), ('monte_carlo', 0), ('lstm', 0), ('arima', 0), ('linear_regression', 0.37209943095992204)]}

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
