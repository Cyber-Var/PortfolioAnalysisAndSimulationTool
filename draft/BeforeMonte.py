import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm


class Calculations:

    # TODO: is it always 252 ?
    trading_year = 252

    def __init__(self, tickers, values, num_of_simulations, start_date, end_date, user_end_date, time_increment):
        self.tickers = tickers
        self.values = np.array(values)
        self.num_of_simulations = num_of_simulations
        self.start_date = start_date
        self.end_date = end_date
        self.user_end_date = user_end_date
        self.time_increment = time_increment

        self.prediction_length = None
        self.close_prices = None
        self.labels = None

        # Retrieve historical data from Yahoo! Finance:
        self.train_set, self.test_set = self.downloadData()
        # S_0:
        self.s_0 = self.train_set.iloc[-1]
        # t:
        self.time_points = self.calculateTimePoints()
        # Daily Returns = R(t) = (P(t) - P(t-1)) / P(t-1) = (P(t) / P(t-1)) - 1
        self.returns = self.train_set.pct_change().dropna()
        # mu:
        self.mu = self.returns.mean()
        # sigma:
        self.sigma = self.returns.std()

    def downloadData(self):
        df = yf.download(self.tickers, start=self.start_date, end=self.user_end_date)
        self.close_prices = df["Adj Close"]
        train_set = self.close_prices.loc[self.start_date:self.end_date]
        test_set = self.close_prices.loc[self.end_date:self.user_end_date]
        return train_set, test_set

    def calculateTimePoints(self):
        days = pd.date_range(start=pd.to_datetime(self.end_date,
                                                  format="%Y-%m-%d") + pd.Timedelta('1 days'),
                             end=pd.to_datetime(self.user_end_date,
                                                format="%Y-%m-%d"))
        weekdays_arr = days.to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0)
        weekdays = weekdays_arr.sum()

        self.prediction_length = int(weekdays / self.time_increment)
        time_points = np.arange(1, self.prediction_length + 1)
        return time_points

    # Method that calculates Geometric Brownian Motion:
    def calculateGBM(self, ticker, drift):
        # Geometric Brownian Motion formula:
        # S_t = S_0 * exp((mu - 0.5 * sigma^2) * time_points + sigma * time_points * Z)
        Z = np.random.normal(0, 1, self.prediction_length).cumsum()
        vol = self.sigma[ticker] * Z
        gbm = self.s_0[ticker] * np.exp(drift + vol)
        return gbm

    def simulateMonteCarlo(self, ticker):
        # Drift:
        drift = (self.mu[ticker] - 0.5 * (self.sigma[ticker] ** 2)) * self.time_points
        # Loop the "calculateGBM" method for Monte Carlo simulation
        monte = np.array([self.calculateGBM(ticker, drift) for _ in range(0, self.num_of_simulations)])

        less = 0
        more = 0
        s_0 = self.s_0[ticker]
        for n in monte:
            m = n[-1]
            if m > s_0:
                more += 1
            elif m < s_0:
                less += 1

        percentage = (max(more, less) / self.num_of_simulations) * 100
        if more >= less:
            prediction = str(percentage) + "% growth"
        else:
            prediction = str(percentage) + "% fall"
        print(prediction)
        return monte

    def plotSimulation(self, ticker, monte):
        # Add starting point to each path:
        monte2 = np.hstack((np.array([[self.s_0[ticker]] for _ in range(self.num_of_simulations)]), monte))

        # Draw the simulation graph:
        for i in range(self.num_of_simulations):
            # plt.plot(monte2[i], alpha=0.5)
            x_axis = pd.date_range(start=self.train_set.index[-1],
                                   end=self.user_end_date,
                                   freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
            plt.plot(x_axis, monte2[i], alpha=0.5)
        plt.ylabel('Price in USD')
        plt.xlabel('Prediction Days')
        plt.xticks(fontsize=9)
        plt.xticks(rotation=340)

        plt.axhline(y=self.s_0[ticker], color='r', linestyle='-')
        # plt.figure(figsize=(6.4, 10))

        _, labels = plt.yticks()
        plt.show()

        prices = []
        for label in labels:
            prices.append(int(label.get_text()))

        return prices

    def printProbabilities(self, ticker, labels, monte):
        start_price = self.s_0[ticker]

        difference = int((labels[1] - labels[0]) / 2)
        for i in range(0, len(labels) - 1):
            labels.append(labels[i] + difference)
        labels.sort()
        prices = np.array(labels)

        less = prices[prices < start_price]
        more = prices[prices > start_price]
        test_data = [mo[-1] for mo in monte]
        results = []
        num = 100 / self.num_of_simulations

        for l in less:
            percentage = (test_data < l).sum() * num
            if percentage > 0:
                results.append("< " + str(l) + ": " + str(percentage))
        for m in more:
            percentage = (test_data > m).sum() * num
            if percentage > 0:
                results.append("> " + str(m) + ": " + str(percentage))
        print(results)

    def getDailyReturns(self, ticker):
        data = self.train_set[ticker]
        daily_returns = data / data.shift(1) - 1
        return daily_returns

    # TODO: can regulate daily/weekly/monthly/annual
    def calculateVolatility(self, ticker):
        daily_returns = self.getDailyReturns(ticker)
        volatility = daily_returns.std() * 100

        if volatility < 2:
            daily_category = "Low"
        elif 2 <= volatility <= 3:
            daily_category = "Normal"
        else:
            daily_category = "High"

        return volatility, daily_category

    def calculateWeights(self):
        total = self.values.sum()
        weights = self.values / total
        return weights

    def calculatePortfolioVolatility(self):
        weights = self.calculateWeights()
        individual_volatilities = np.array([self.calculateVolatility(ticker)[0] / 100 for ticker in self.tickers])

        variance = np.dot(weights ** 2, individual_volatilities ** 2)
        portfolio_volatility = np.sqrt(variance) * 100

        if portfolio_volatility < 2:
            daily_category = "Low"
        elif 2 <= portfolio_volatility <= 3:
            daily_category = "Normal"
        else:
            daily_category = "High"

        return "Portfolio volatility: " + daily_category + " (" + str(portfolio_volatility) + ")"

    def calculateSharpeRatio(self, ticker):
        daily_returns = self.getDailyReturns(ticker)

        risk_free_rate = 1.02 ** (1 / 252) - 1
        excess = daily_returns - risk_free_rate
        sharpe_ratio = excess.mean() / excess.std()

        if sharpe_ratio < 0.1:
            sharpe_category = "Low"
        elif 0.1 <= sharpe_ratio <= 0.2:
            sharpe_category = "Normal"
        else:
            sharpe_category = "High"
        return "Sharpe Ratio: " + sharpe_category + " (" + str(sharpe_ratio) + ")"

    # TODO: can regulate daily/weekly/monthly/annual
    def calculateVaR(self, ticker, confidence):
        today = date.today().strftime("%Y-%m-%d")
        if self.user_end_date > today:
            # current = today
            portfolio_value = self.close_prices[ticker].iloc[-1]
        else:
            # current = self.end_date
            portfolio_value = self.train_set[ticker].iloc[-1]
        Z = norm.ppf(1 - confidence)
        VaR = portfolio_value * self.calculateVolatility(ticker)[0] * Z
        return VaR



calc = Calculations(["AAPL", "TSLA", "MSFT"], [2000, 10000, 1000], 10000,
                    '2023-01-01', '2023-08-29', "2023-09-20", 1)
ticker = "TSLA"
# monte = calc.simulateMonteCarlo(ticker)

'''m275 = 0
m300 = 0
m325 = 0
m350 = 0
m375 = 0

l250 = 0
l225 = 0
l200 = 0
l175 = 0
l150 = 0

for i in monte:
    test = i[-1]
    if test > 275:
        m275 += 1
    if test > 300:
        m300 += 1
    if test > 325:
        m325 += 1
    if test > 350:
        m350 += 1
    if test > 375:
        m375 += 1

    if test < 250:
        l250 += 1
    if test < 225:
        l225 += 1
    if test < 200:
        l200 += 1
    if test < 175:
        l175 += 1
    if test < 150:
        l150 += 1

print("> 275:", m275 / 100)
print("> 300:", m300 / 100)
print("> 325:", m325 / 100)
print("> 350:", m350 / 100)
print("> 375:", m375 / 100)
print()
print("< 250:", l250 / 100)
print("< 225:", l225 / 100)
print("< 200:", l200 / 100)
print("< 175:", l175 / 100)
print("< 150:", l150 / 100)'''

# labels = calc.plotSimulation(ticker, monte)
# calc.printProbabilities(ticker, labels, monte)
vol, cat = calc.calculateVolatility(ticker)
print("Volatility:", cat, "(" + str(vol) + ")")
print(calc.calculatePortfolioVolatility())
print(calc.calculateSharpeRatio(ticker))
print("VaR: " + str(calc.calculateVaR(ticker, 0.95)))

# User input:
'''num_of_simulations = 10000
start_date = '2023-02-20'
end_date = '2023-08-20'
tickers = ["AAPL", "TSLA"]
user_end_date = "2024-01-04"
time_increment = 1
trading_year = 252'''

'''less = 0
less2 = 0
more = 0
eq = 0
test = history_data[-1]
print(test)
for n in monte:
    for m in n:
        if m < test:
            less += 1
        elif m > test:
            more += 1
        else:
            eq += 1

        if m <= 150:
            less2 += 1
print("<", test, less)
print(">", test, more)
print("==", test, eq)
print("less2", less2)'''
