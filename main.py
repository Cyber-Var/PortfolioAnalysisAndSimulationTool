import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from MonteCarlo import MonteCarloSimulation
from RiskMetrics import RiskMetrics
from RandomForest import RandomForestRegressionAlgorithm
from LinearRegression import LinearRegressionAlgorithm


class Calculations:

    # TODO: is it always 252 ?
    trading_year = 252

    def __init__(self, tickers, investments, num_of_simulations, start_date, end_date, user_end_date, time_increment):
        self.tickers = tickers
        self.investments = np.array(investments)
        self.num_of_simulations = num_of_simulations
        self.start_date = start_date
        self.end_date = end_date
        self.user_end_date = user_end_date
        self.time_increment = time_increment

        self.prediction_length = None
        # self.close_prices = None
        self.labels = None

        # Retrieve historical data from Yahoo! Finance:
        self.data = self.downloadData()
        self.history_data = self.data[:end_date]
        # self.close_prices = self.getHistoryCloseData()

        # Create a list of dates that includes weekdays only:
        self.weekdays = self.getWeekDays()

        # TODO: re-make it into loop that goes through each share in portfolio
        apple_data = self.getDataForTicker("AAPL", self.history_data)
        end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        apple_future_data = self.getDataForTicker("AAPL", self.data)[end:]

        # Monte Carlo Simulation:
        # monte = MonteCarloSimulation(investments, num_of_simulations, end_date, user_end_date, time_increment,
        #                              apple_data["Adj Close"], self.weekdays)

        # Calculate risk metrics:
        # risk = RiskMetrics(tickers, self.investments, user_end_date, "TSLA", self.history_data["Adj Close"])

        # Random Forest Regression Algorithm:
        random_forest = RandomForestRegressionAlgorithm(end_date, user_end_date, apple_data, apple_future_data)

        # Linear Regression Algorithm:
        # linear = LinearRegressionAlgorithm(end_date, user_end_date, apple_data, apple_future_data)

    def downloadData(self):
        data = yf.download(self.tickers, start=self.start_date,
                           end=(datetime.strptime(self.user_end_date, "%Y-%m-%d") +
                                timedelta(days=1)).strftime("%Y-%m-%d"))
        return data

    def getDataForTicker(self, ticker, data):
        ticker_data = pd.DataFrame()
        for col, ti in data.columns:
            ticker_data[col] = data[col][ticker]
        return ticker_data

    def getWeekDays(self):
        weekdays = pd.bdate_range(start=pd.to_datetime(self.end_date, format="%Y-%m-%d") + pd.Timedelta('1 days'),
                                  end=pd.to_datetime(self.user_end_date, format="%Y-%m-%d"))
        return weekdays


calc = Calculations(["AAPL", "TSLA", "MSFT"], [2000, 10000, 1000], 10000,
                    '2023-01-01', '2023-11-25', "2023-12-04", 1)
