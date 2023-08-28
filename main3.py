'''import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

start_date = '2022-09-19'
end_date = '2022-12-01'
ticker_symbol = 'TSLA'

data = yf.download(ticker_symbol, start=start_date, end=end_date)
data['Daily_Return'] = data['Adj Close'].pct_change().dropna()

mean_return = data['Daily_Return'].mean()
std_dev_return = data['Daily_Return'].std()

initial_price = data['Adj Close'].iloc[-1]  # Use the last available price as the starting point
time_to_predict = (datetime(2023, 1, 4) - datetime(2022, 12, 1)).days / 365.0

result = []
for i in range(0, 10000):
    Z = np.random.normal(0, 1)
    predicted_price = initial_price * np.exp((mean_return - 0.5 * std_dev_return**2) * time_to_predict + std_dev_return * np.sqrt(time_to_predict) * Z)
    result.append(predicted_price)

less = 0
less2 = 0
more = 0
equal = 0
for n in result:
    if n < 194.7:
        less += 1
    elif n > 194.7:
        more += 1
    else:
        equal += 1

    if n <= 150:
        less2 += 1

print("less", less, "more", more, "equal", equal, "less2", less2)

print(f"Predicted Tesla stock price on 4 January 2023: ${predicted_price:.2f}")'''

import yfinance as yf
import numpy as np

# Define the ticker symbol and date range
ticker_symbol = "TSLA"
start_date = '2022-09-19'
end_date = '2022-12-01'

# Fetch historical data from Yahoo Finance
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Calculate daily returns using pct_change
data['Return'] = data['Adj Close'].pct_change()

# Calculate average return
average_return = data['Return'].mean()

# Calculate squared deviations from the average return
data['Squared_Deviation'] = (data['Return'] - average_return)**2

# Calculate sample variance
sample_variance = data['Squared_Deviation'].sum() / (len(data) - 1)

# Calculate historical volatility (standard deviation)
historical_volatility = np.sqrt(sample_variance)

print("Historical Volatility:", historical_volatility)
