import numpy as np
from scipy.stats import norm


class RiskMetrics:

    def __init__(self, tickers, investments, close_prices):
        self.tickers = tickers
        self.investments = investments
        self.close_prices = close_prices

        self.daily_returns = self.getDailyReturns()

    def getDailyReturns(self):
        data = self.close_prices
        daily_returns = data / data.shift(1) - 1
        return daily_returns

    # TODO: can regulate daily/weekly/monthly/annual
    def calculateVolatility(self, ticker):
        if len(self.tickers) > 1:
            volatility = self.daily_returns[ticker].std() * 100
        else:
            volatility = self.daily_returns.std() * 100

        if volatility < 2:
            daily_category = "Low"
        elif 2 <= volatility <= 3:
            daily_category = "Normal"
        else:
            daily_category = "High"

        return volatility, daily_category

    def calculateWeights(self):
        invs = np.array(self.investments)
        total = invs.sum()
        weights = invs / total
        return weights

    # TODO: fix because it will break as overall data in main is different
    def calculatePortfolioVolatility(self):
        weights = self.calculateWeights()
        individual_volatilities = np.array([self.calculateVolatility(t)[0] / 100 for t in self.tickers])

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
        risk_free_rate = 1.02 ** (1 / 252) - 1
        if len(self.tickers) > 1:
            excess = self.daily_returns[ticker] - risk_free_rate
        else:
            excess = self.daily_returns - risk_free_rate
        sharpe_ratio = excess.mean() / excess.std()

        if sharpe_ratio < 0.1:
            sharpe_category = "Low"
        elif 0.1 <= sharpe_ratio <= 0.2:
            sharpe_category = "Normal"
        else:
            sharpe_category = "High"
        return "Sharpe Ratio: " + sharpe_category + " (" + str(sharpe_ratio) + ")"

    # TODO: can regulate daily/weekly/monthly/annual
    def calculateVaR(self, ticker, confidence, volatility):
        if len(self.tickers) > 1:
            portfolio_value = self.close_prices[ticker].iloc[-1]
        else:
            portfolio_value = self.close_prices.iloc[-1]
        Z = norm.ppf(1 - confidence)
        VaR = portfolio_value * volatility * Z
        return VaR
