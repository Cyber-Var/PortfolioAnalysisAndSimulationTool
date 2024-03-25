from datetime import date

import numpy as np
from scipy.stats import norm
from dateutil.relativedelta import relativedelta


class RiskMetrics:

    def __init__(self, tickers, investments, longs_shorts, close_prices):
        self.tickers = list(tickers)
        self.investments = list(investments)
        self.close_prices = close_prices
        self.longs_shorts = [1 if x else -1 for x in longs_shorts]

        self.daily_returns = self.getDailyReturns()
        self.today = date.today()

    def getDailyReturns(self):
        # daily_returns = self.close_prices / data.shift(1) - 1
        return self.close_prices.pct_change()

    # TODO: can regulate daily/weekly/monthly/annual
    def calculateVolatility(self, ticker):
        daily_returns = self.close_prices[(self.today - relativedelta(months=6)):].pct_change()
        if len(self.tickers) > 1:
            volatility = daily_returns[ticker].std() * 100
        else:
            volatility = daily_returns.std() * 100

        if volatility < 2:
            daily_category = "Low"
        elif 2 <= volatility <= 3:
            daily_category = "Normal"
        else:
            daily_category = "High"

        return volatility, daily_category

    def calculateSharpeRatio(self, ticker):
        risk_free_rate = 1.02 ** (1 / 252) - 1
        if len(self.tickers) == 1:
            excess = self.daily_returns - risk_free_rate
        else:
            excess = self.daily_returns[ticker] - risk_free_rate
        sharpe_ratio = excess.mean() / excess.std()

        if sharpe_ratio < 0.1:
            sharpe_category = "Low"
        elif 0.1 <= sharpe_ratio <= 0.2:
            sharpe_category = "Normal"
        else:
            sharpe_category = "High"
        return sharpe_ratio, sharpe_category

    # TODO: can regulate daily/weekly/monthly/annual
    # TODO: choose 1 of the methods below (choice is between historical VaR and Parametric VaR)
    def calculateVaR(self, ticker, confidence, volatility):
        ticker_index = self.tickers.index(ticker)

        # if len(self.tickers) > 1:
        #     portfolio_value = self.close_prices[ticker].iloc[-1]
        # else:
        #     portfolio_value = self.close_prices.iloc[-1]
        # Z = norm.ppf(1 - confidence)
        # VaR = portfolio_value * volatility * Z

        if len(self.tickers) == 1:
            daily_returns = self.daily_returns
        else:
            daily_returns = self.daily_returns[ticker]
        historical_losses = [r for r in daily_returns if r < 0]
        historical_losses.sort()

        index = int(len(historical_losses) * (1 - confidence))
        percentile_loss = historical_losses[index]
        VaR = -1 * percentile_loss * self.investments[ticker_index]

        return VaR

    def calculatePortfolioWeights(self):
        adjusted_investments = np.array(self.investments) * np.array(self.longs_shorts)
        portfolio_total = np.sum(np.abs(adjusted_investments))
        weights = adjusted_investments / portfolio_total
        return weights

    def calculatePortfolioVolatility(self):
        # individual_volatilities = np.array([self.calculateVolatility(t)[0] / 100 for t in self.tickers])
        weights = self.calculatePortfolioWeights()

        # if len(weights.shape) == 1:
        #     weights = weights.reshape(-1, 1)

        close_prices = self.close_prices[(self.today - relativedelta(months=6)):]
        if len(self.close_prices.values.shape) == 1:
            close_prices = close_prices.values.reshape(-1, 1)
        else:
            close_prices = close_prices.values

        daily_returns = np.diff(close_prices, axis=0) / close_prices[:-1, :]
        covariance_matrix = np.cov(daily_returns.T)

        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance) * 100

        if portfolio_volatility < 2:
            category = "Low"
        elif 2 <= portfolio_volatility <= 3:
            category = "Normal"
        else:
            category = "High"

        return portfolio_volatility, category

    def calculatePortfolioSharpeRatio(self, portfolio_vol):
        risk_free_rate = 1.02 ** (1 / 252) - 1
        weights = self.calculatePortfolioWeights()

        expected_returns = np.mean(self.daily_returns, axis=0)

        if len(self.tickers) == 1:
            portfolio_expected_returns = expected_returns
        else:
            portfolio_expected_returns = 0
            for i in range(len(weights)):
                portfolio_expected_returns += weights[i] * expected_returns.iloc[i]

        portfolio_sharpe_ratio = (portfolio_expected_returns - risk_free_rate) / (portfolio_vol / 100)

        if abs(portfolio_sharpe_ratio) < 0.1:
            category = "Low"
        elif 0.1 <= abs(portfolio_sharpe_ratio) <= 0.2:
            category = "Normal"
        else:
            category = "High"

        return abs(portfolio_sharpe_ratio), category

    def calculatePortfolioVaR(self, hold_duration, confidence, portfolio_vol):
        weights = self.calculatePortfolioWeights()
        total_investment_amount = sum(self.investments)

        # avg_rets = daily_returns.mean()
        # port_mean = avg_rets.dot(weights)
        # mean_investment = (1 + port_mean) * total_investment_amount
        # stdev_investment = total_investment_amount * portfolio_vol
        # Z = norm.ppf(1 - confidence, mean_investment, stdev_investment)
        # portfolio_VaR = total_investment_amount - Z

        if len(self.tickers) > 1:
            weighted_returns = (self.daily_returns * weights).sum(axis=1)
            historical_losses = weighted_returns[weighted_returns < 0].sort_values()

            index = int((1 - confidence) * len(historical_losses))
            percentile_loss = historical_losses.iloc[index]
        else:
            weighted_returns = (self.daily_returns * weights).sum()
            percentile_loss = weighted_returns

        portfolio_VaR = -1 * percentile_loss * total_investment_amount

        if hold_duration == "1d":
            num_days = 1
        elif hold_duration == "1w":
            num_days = 5
        else:
            num_days = 20

        return abs(np.round(portfolio_VaR * np.sqrt(num_days), 2))

