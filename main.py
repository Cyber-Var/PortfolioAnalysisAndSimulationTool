import yfinance as yf
import pandas as pd
import numpy as np

from MonteCarlo import MonteCarloSimulation
from RiskMetrics import RiskMetrics
from RandomForest import RandomForestAlgorithm


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
        self.history_data = self.downloadData()
        self.close_prices = self.getHistoryCloseData()

        # Create a list of dates that includes weekdays only:
        self.weekdays = self.getWeekDays()

        # Monte Carlo Simulation:
        # monte = MonteCarloSimulation(investments, num_of_simulations, end_date, user_end_date, time_increment,
        #                              self.close_prices["AAPL"], self.weekdays)

        # Calculate risk metrics:
        # risk = RiskMetrics(tickers, self.investments, user_end_date, "TSLA", self.close_prices)

        # Random Forest Algorithm:
        random_forest = RandomForestAlgorithm(end_date, "AAPL", self.history_data)

    def downloadData(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.user_end_date)
        return data

    def getHistoryCloseData(self):
        close_prices = self.history_data["Adj Close"]
        close_prices = close_prices.loc[self.start_date:self.end_date]
        return close_prices

    def getWeekDays(self):
        weekdays = pd.bdate_range(start=pd.to_datetime(self.end_date, format="%Y-%m-%d") + pd.Timedelta('1 days'),
                                  end=pd.to_datetime(self.user_end_date, format="%Y-%m-%d"))
        return weekdays


calc = Calculations(["AAPL", "TSLA", "MSFT"], [2000, 10000, 1000], 10000,
                    '2023-01-01', '2023-08-29', "2023-09-20", 1)
