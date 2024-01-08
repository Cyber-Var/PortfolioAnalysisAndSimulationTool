import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

from MonteCarlo import MonteCarloSimulation
from RiskMetrics import RiskMetrics
from RandomForest import RandomForestRegressionAlgorithm
from LinearRegressionAlgorithm import LinearRegressionAlgorithm


class Model:
    # TODO: is it always 252 ?
    trading_year = 252

    def __init__(self, tickers, investments, num_of_simulations, hold_duration, time_increment):
        self.tickers = tickers
        self.investments = np.array(investments)
        self.num_of_simulations = num_of_simulations
        self.hold_duration = hold_duration
        self.time_increment = time_increment

        today = date.today()
        self.end_date = today

        # TODO: choose historical date range here:
        if hold_duration == "1d":
            self.start_date = today - relativedelta(months=6)
            self.prediction_date = today + relativedelta(days=1)
        elif hold_duration == "1w":
            self.start_date = today - relativedelta(years=1)
            self.prediction_date = today + relativedelta(days=7)
        elif hold_duration == "1m":
            self.start_date = today - relativedelta(years=3)
            self.prediction_date = today + relativedelta(months=1)
        elif hold_duration == "3m":
            self.start_date = today - relativedelta(years=3)
            self.prediction_date = today + relativedelta(months=3)

        # self.prediction_length = None
        # self.labels = None

        # Retrieve historical data from Yahoo! Finance:
        self.data = self.downloadData()
        # self.history_data = self.data[:end_date]

        if hold_duration == "1d" and ((today.weekday() == 4 and self.data.index[-1] == today.strftime('%Y-%m-%d'))
                                      or today.weekday() == 5):
            print("Prediction = ", self.data["Adj Close"])
            # TODO: if this is the case, also make Monte Carlo graph into horizontal line
        else:
            # TODO: re-make it into loop that goes through each share in portfolio
            apple_data = self.getDataForTicker("AAPL", self.data)
            if len(apple_data) < 30:
                raise Exception("Unable to predict - the share was created too recently.")
            else:
                # Linear Regression Algorithm:
                linear = LinearRegressionAlgorithm(hold_duration, apple_data, self.prediction_date)

                # Random Forest Regression Algorithm:
                random_forest = RandomForestRegressionAlgorithm(hold_duration, apple_data, self.prediction_date)

                # Monte Carlo Simulation:
                # Create a list of dates that includes weekdays only:
                # self.weekdays = self.getWeekDays()
                # monte = MonteCarloSimulation(investments, num_of_simulations, self.prediction_date, time_increment,
                #                              apple_data["Adj Close"], self.weekdays, hold_duration)

        # Calculate risk metrics:
        risk = RiskMetrics(tickers, self.investments, "TSLA", self.data["Adj Close"])

    def downloadData(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
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
        elif self.hold_duration == "1m":
            weekdays = pd.date_range(self.prediction_date - relativedelta(months=1) + relativedelta(days=1),
                                     self.prediction_date,
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        else:
            weekdays = pd.date_range(self.prediction_date - relativedelta(months=3) + relativedelta(days=1),
                                     self.prediction_date,
                                     freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        return weekdays


calc = Model(["AAPL", "TSLA", "MSFT"], [2000, 10000, 1000], 10000,
             "3m", 1)
