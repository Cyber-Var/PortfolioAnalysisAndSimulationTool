import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

from MonteCarlo import MonteCarloSimulation
from RiskMetrics import RiskMetrics
from RandomForest import RandomForestAlgorithm
from LinearRegressionAlgorithm import LinearRegressionAlgorithm
from EthicalScore import ESGScores


from LSTM import LSTMAlgorithm


class Model:

    def __init__(self, tickers, investments, num_of_simulations, hold_duration, time_increment):
        self.tickers = tickers
        self.investments = np.array(investments)
        self.num_of_simulations = num_of_simulations
        self.hold_duration = hold_duration
        self.time_increment = time_increment

        today = date.today()
        end_date = today

        # TODO: choose historical date range here using user's preferred investment behaviour:
        if hold_duration == "1d":
            start_date = today - relativedelta(months=18)
            self.prediction_date = today + relativedelta(days=1)
        elif hold_duration == "1w":
            start_date = today - relativedelta(years=2)
            self.prediction_date = today + relativedelta(days=7)
        else:
            start_date = today - relativedelta(years=4)
            self.prediction_date = today + relativedelta(months=1)

        # Retrieve historical data from Yahoo! Finance:
        self.data = self.downloadData(start_date, end_date)

        is_flat_monte_graph = False
        if hold_duration == "1d" and ((today.weekday() == 4 and self.data.index[-1] == today.strftime('%Y-%m-%d'))
                                      or today.weekday() == 5):
            print("Prediction = ", self.data["Adj Close"])
            is_flat_monte_graph = True
        else:
            # TODO: re-make it into loop that goes through each share in portfolio
            apple_data = self.getDataForTicker("AAPL", self.data)
            if len(apple_data) < 100:
                raise Exception("Unable to predict - the share was created too recently.")
            else:
                # esg = ESGScores(self.tickers)

                # lstm = LSTMAlgorithm(hold_duration, apple_data, self.prediction_date, start_date)

                # Linear Regression Algorithm:
                # linear = LinearRegressionAlgorithm(hold_duration, apple_data, self.prediction_date, start_date)

                # Random Forest Regression Algorithm:
                # random_forest = RandomForestAlgorithm(hold_duration, apple_data, self.prediction_date, start_date)

                # Monte Carlo Simulation:
                # Create a list of dates that includes weekdays only:
                self.weekdays = self.getWeekDays()
                monte = MonteCarloSimulation(investments, num_of_simulations, self.prediction_date,
                                             apple_data["Adj Close"], self.weekdays, hold_duration, start_date)

        # Calculate risk metrics:
        # risk = RiskMetrics(tickers, self.investments, "TSLA", self.data["Adj Close"])

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


calc = Model(["AAPL", "TSLA", "MSFT"], [2000, 10000, 1000], 10000,
             "1d", 1)
