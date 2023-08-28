from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import date
from scipy.stats import norm


# User input:
num_of_simulations = 10000
start_date = '2023-02-20'
end_date = '2023-08-20'
tickers = "AAPL"
user_end_date = "2024-01-04"
time_increment = 1
trading_year = 252


df = yf.download(tickers, start=start_date, end=user_end_date)
close_prices = df["Adj Close"]
history_data = close_prices.loc[start_date:end_date]

# Geometric Brownian Motion formula:
# S_t = S_0 * exp((mu - 0.5 * sigma^2) * time_points + sigma * time_points * Z)

# S_0:
s_0 = history_data[-1]

# t:
days = pd.date_range(start=pd.to_datetime(end_date,
                                          format="%Y-%m-%d") + pd.Timedelta('1 days'),
                     end=pd.to_datetime(user_end_date,
                                        format="%Y-%m-%d"))
weekdays_arr = days.to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0)
weekdays = weekdays_arr.sum()

prediction_length = int(weekdays / time_increment)
time_points = np.arange(1, prediction_length + 1)

# Daily Returns = R(t) = (P(t) - P(t-1)) / P(t-1) = (P(t) / P(t-1)) - 1
returns = history_data.pct_change().dropna()

# mu and sigma:
mu = returns.mean()
sigma = returns.std()

# Drift:
drift = (mu - 0.5 * (sigma ** 2)) * time_points


# Method that calculates Geometric Brownian Motion:
def calculateGBM():
    Z = np.random.normal(0, 1, prediction_length).cumsum()
    vol = sigma * Z
    gbm = s_0 * np.exp(drift + vol)
    return gbm


# TODO: uncomment
# Loop the "calculateGBM" method for Monte Carlo simulation
monte = np.array([calculateGBM() for i in range(0, num_of_simulations)])

# Add starting point to each path:
monte2 = np.hstack((np.array([[s_0] for j in range(num_of_simulations)]), monte))

# Draw the simulation graph:
for i in range(num_of_simulations):
    plt.plot(monte2[i], alpha=0.5)

plt.axhline(y=history_data[-1], color='r', linestyle='-')
plt.show()

daily_returns = history_data / history_data.shift(1) - 1


# TODO: can regulate daily/weekly/monthly/annual
def calculateVolatility():
    volatility = daily_returns.std() * 100

    if volatility < 2:
        daily_category = "Low"
    elif 2 <= volatility <= 3:
        daily_category = "Normal"
    else:
        daily_category = "High"

    # return "Volatility:", daily_category, "(" + str(volatility) + ")"
    return volatility


def calculate_sharpe_ratio():
    risk_free_rate = 1.02 ** (1 / 252) - 1
    excess = daily_returns - risk_free_rate
    sharpe_ratio = excess.mean() / excess.std()

    if sharpe_ratio < 0.1:
        sharpe_category = "Low"
    elif 0.1 <= sharpe_ratio <= 0.2:
        sharpe_category = "Normal"
    else:
        sharpe_category = "High"
    return "Sharpe Ratio:", sharpe_category, "(" + str(sharpe_ratio) + ")"


# TODO: VaR = portfolio_value * portfolio_volatility
# TODO: can regulate confidence level
# for now (for testing): confidence level = 0.95
def calculateVaR(confidence):
    today = date.today().strftime("%Y-%m-%d")

    if user_end_date > today:
        # current = today
        portfolio_value = close_prices[-1]
    else:
        # current = end_date
        portfolio_value = history_data[-1]

    '''Z = norm.ppf(1 - confidence)

    # TODO: Later this should be proportion of each asset's market value in the portfolio
    # so they sum up to 1
    # The ones below will be lists with elements for each ticker
    proportion = np.array([1])
    return_means = np.array([returns.mean()])
    expected_returns = np.dot(return_means, proportion)

    VaR = -1 * portfolio_value * Z * sigma + portfolio_value * expected_returns'''


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
