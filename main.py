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
# from LSTM import LSTMAlgorithm
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage
from UI.PortfolioPage import PortfolioPage

import logging


class Controller:

    def __init__(self, tickers, investments, num_of_simulations, hold_duration, time_increment):
        self.tickers = tickers
        self.investments = np.array(investments)
        self.num_of_simulations = num_of_simulations
        self.hold_duration = hold_duration
        self.time_increment = time_increment

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        today = date.today()
        end_date = today

        # TODO: choose historical date range here using user's preferred investment behaviour:
        if hold_duration == "1d":
            start_date = today - relativedelta(months=18)
            self.prediction_date = today + relativedelta(days=1)
            while self.prediction_date.weekday() >= 5:
                self.prediction_date = self.prediction_date + relativedelta(days=1)
            self.moving_avg_value = 5
        elif hold_duration == "1w":
            start_date = today - relativedelta(years=2)
            self.prediction_date = today + relativedelta(days=7)
            self.moving_avg_value = 20
        else:
            start_date = today - relativedelta(years=4)
            self.prediction_date = today + relativedelta(months=1)
            self.moving_avg_value = 80

        # Retrieve historical data from Yahoo! Finance:
        self.data = self.downloadData(start_date, end_date)

        is_flat_monte_graph = False
        # if hold_duration == "1d" and ((today.weekday() == 4 and self.data.index[-1] == today.strftime('%Y-%m-%d'))
        #                               or today.weekday() == 5):
        #     print("Prediction = ", self.data["Adj Close"])
        #     is_flat_monte_graph = True
        # else:
        # TODO: re-make it into loop that goes through each share in portfolio
        apple_data = self.getDataForTicker("AAPL", self.data)
        if len(apple_data) < 100:
            raise Exception("Unable to predict - the share was created too recently.")
        else:
            # self.plotMovingAverage(apple_data, "AAPL")

            # ARIMA:
            # aapl = yf.Ticker("AAPL")
            # today_data = aapl.history(period="1d")
            # arima = ARIMAAlgorithm(hold_duration, apple_data, self.prediction_date, start_date, today_data, [20])
            # print("ARIMA Evaluation:")
            # mse, rmse, mae, mape, r2 = arima.evaluateModel()
            # arima.printEvaluation(mse, rmse, mae, mape, r2)
            # data_for_prediction = arima.get_data_for_prediction()
            # predictions = arima.predict_price(data_for_prediction)
            # arima.plot_arima(predictions, data_for_prediction)

            # ESG Scores:
            # esg = ESGScores(self.tickers)

            # LSTM:
            # lstm = LSTMAlgorithm(hold_duration, apple_data, self.prediction_date, start_date, (3, 50, 0.2, 25,
            #                                                                                    'adam',
            #                                                                                    'mean_squared_error'))
            # mse, rmse, mae, mape, r2 = lstm.evaluateModel()
            # print("LSTM Evaluation:")
            # lstm.printEvaluation(mse, rmse, mae, mape, r2)
            # prediction = lstm.predict_price(lstm.get_data_for_prediction())

            # Linear Regression Algorithm:
            # linear_regression = LinearRegressionAlgorithm(hold_duration, apple_data, self.prediction_date, start_date,
            #                                               (True,))
            # self.run_model(linear_regression, "Linear Regression")

            # Random Forest Regression Algorithm:
            # random_forest = RandomForestAlgorithm(hold_duration, apple_data, self.prediction_date, start_date, (50,
            #                                                                                                     'sqrt', 5, 2, 1, True, "squared_error", None))
            # self.run_model(random_forest, "Random Forest Regression")

            # Bayesian Regression Algorithm:
            # bayesian = BayesianRegressionAlgorithm(hold_duration, apple_data, self.prediction_date, start_date, (100,
            #                                                                                                      1e-3, 1e-6, 1e-6, 1e-6, 1e-6, True, True, True))
            # self.run_model(bayesian, "Bayesian Ridge Regression")

            # Monte Carlo Simulation:
            # Create a list of dates that includes weekdays only:
            self.weekdays = self.getWeekDays()
            # monte = MonteCarloSimulation(num_of_simulations, self.prediction_date,
            #                              apple_data["Adj Close"], self.weekdays, hold_duration, start_date)
            # print("Monte Carlo Simulation Evaluation:")
            # mse, rmse, mae, mape, r2 = monte.evaluateModel()
            # monte.printEvaluation(mse, rmse, mae, mape, r2)
            # results, s_0 = monte.makeMCPrediction(monte.get_data_for_prediction())
            # monte.displayResults(results, s_0)

            # parameter_tester = ParameterTester(hold_duration, apple_data, self.prediction_date, start_date,
            #                                    num_of_simulations, self.weekdays)

            # app = QApplication(sys.argv)
            # menu_page = MenuPage()
            # menu_page.show()
            # sys.exit(app.exec_())

            app = QApplication(sys.argv)
            main_window = MainWindow()
            menu_page = MenuPage(main_window)
            main_window.setCentralWidget(menu_page)
            main_window.show()
            sys.exit(app.exec_())

        # Calculate risk metrics:
        # risk = RiskMetrics(tickers, self.investments, "TSLA", self.data["Adj Close"])

    def run_model(self, model, model_name):
        mse, rmse, mae, mape, r2 = model.evaluateModel()
        print(model_name, "Evaluation:")
        model.printEvaluation(mse, rmse, mae, mape, r2)
        model.predict_price()

    def downloadData(self, start, end):
        data = yf.download(self.tickers, start=start, end=end)
        return data

    def getDataForTicker(self, ticker, data):
        ticker_data = pd.DataFrame()
        for col, ti in data.columns:
            ticker_data[col] = data[col][ticker]
        return ticker_data

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


calc = Controller(["AAPL", "TSLA", "MSFT"], [2000, 10000, 1000], 1000,
             "1m", 1)
