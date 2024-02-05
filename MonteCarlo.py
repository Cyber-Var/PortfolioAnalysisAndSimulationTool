from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import AutoLocator


class MonteCarloSimulation:

    def __init__(self, values, num_of_simulations, prediction_date, data, num_weekdays,
                 hold_duration, start_date):
        self.values = np.array(values)
        self.num_of_simulations = num_of_simulations
        self.prediction_date = prediction_date
        self.num_weekdays = num_weekdays
        self.hold_duration = hold_duration
        self.start_date = start_date
        self.data = data
        self.time_increment = 1

        self.prediction_length = None
        self.s_0 = None
        self.mu = None
        self.sigma = None

        self.days = 1
        if self.hold_duration == "1w":
            self.days = 5
        elif self.hold_duration == "1m":
            self.days = 20

        # if is_flat:
        #     price = data.iloc[-1]
        #     print("Price will stay the same:", price, "\n")
        #
        #     monte_results = [price, price]
        #     x_axis = [data.index[-1], self.prediction_date]
        #     for i in range(self.num_of_simulations):
        #         plt.plot(x_axis, monte_results, alpha=0.5)
        #     plt.ylabel('Price in USD')
        #     plt.xlabel('Prediction Days')
        #     plt.xticks(fontsize=9)
        #     plt.xticks(rotation=340)
        #     plt.xticks(x_axis)
        #     plt.show()
        # else:
            # self.prediction_length = 0
            #
            # # S_0:
            # self.s_0 = data.iloc[-1]
            # # Daily Returns = R(t) = (P(t) - P(t-1)) / P(t-1) = (P(t) / P(t-1)) - 1
            # # TODO: decide to use daily_returns or log_returns:
            # daily_returns = data.pct_change(self.days).dropna()
            # returns = np.log(1 + daily_returns)
            # # mu:
            # self.mu = returns.mean()
            # # sigma:
            # self.sigma = returns.std()

        # Display Monte Carlo results:
        if self.hold_duration == "1d":
            train_start = date.today() - relativedelta(months=6)
        elif self.hold_duration == "1w":
            train_start = date.today() - relativedelta(years=1)
        else:
            train_start = date.today() - relativedelta(years=3)
        dataset = data[train_start:]

        self.prepareForMC(dataset)
        monte_results = self.simulateMonteCarlo()
        plot_labels = self.plotSimulation(monte_results)
        self.printProbabilities(plot_labels, monte_results)
        # self.evaluateMC()

    def prepareForMC(self, data):
        self.prediction_length = 0

        # S_0:
        self.s_0 = data.iloc[-1]
        # Daily Returns = R(t) = (P(t) - P(t-1)) / P(t-1) = (P(t) / P(t-1)) - 1
        # TODO: decide to use daily_returns or log_returns:
        daily_returns = data.pct_change(self.days).dropna()
        returns = np.log(1 + daily_returns)
        # mu:
        self.mu = returns.mean()
        # sigma:
        self.sigma = returns.std()

    def calculateTimePoints(self):
        self.prediction_length = int(self.num_weekdays / self.time_increment)
        time_points = np.arange(1, self.prediction_length + 1)
        return time_points

    # Method that calculates Geometric Brownian Motion:
    def calculateGBM(self, drift):
        # Geometric Brownian Motion formula:
        # S_t = S_0 * exp((mu - 0.5 * sigma^2) * time_points + sigma * time_points * Z)
        Z = np.random.normal(0, 1, self.prediction_length).cumsum()
        vol = self.sigma * Z
        gbm = self.s_0 * np.exp(drift + vol)
        return gbm

    def simulateMonteCarlo(self):
        # t:
        time_points = self.calculateTimePoints()

        # Drift:
        drift = (self.mu - 0.5 * (self.sigma ** 2)) * time_points
        # Loop the "calculateGBM" method for Monte Carlo simulation
        monte = np.array([self.calculateGBM(drift) for _ in range(0, self.num_of_simulations)])

        less = 0
        more = 0
        for n in monte:
            m = n[-1]
            if m > self.s_0:
                more += 1
            elif m < self.s_0:
                less += 1

        percentage = (max(more, less) / self.num_of_simulations) * 100
        if more >= less:
            prediction = str(percentage) + "% chance of growth"
        else:
            prediction = str(percentage) + "% chance of fall"
        print(prediction)
        return monte

    def plotSimulation(self, monte):
        # Add starting point to each path:
        monte2 = np.hstack((np.array([[self.s_0] for _ in range(self.num_of_simulations)]), monte))

        x_axis = pd.date_range(start=date.today(), end=self.prediction_date,
                               freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()

        # Draw the simulation graph:
        for i in range(self.num_of_simulations):
            plt.plot(x_axis, monte2[i], alpha=0.5)

        plt.ylabel('Price in USD')
        plt.xlabel('Prediction Days')
        plt.xticks(fontsize=9)
        plt.xticks(rotation=340)

        if self.hold_duration == "1d" or self.hold_duration == "1w":
            plt.xticks(x_axis)

        plt.axhline(y=self.s_0, color='r', linestyle='-')

        _, labels = plt.yticks()
        plt.show()

        prices = []
        for label in labels:
            try:
                prices.append(float(label.get_text()))
            except Exception:
                continue

        return prices

    def printProbabilities(self, labels, monte):
        start_price = self.s_0

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
        print(results, "\n")

    # def evaluateMC(self):
    #     # TODO: explain this sliding method clearly in report
    #
    #     all_predictions = []
    #     all_tests = []
    #
    #     train_start = self.start_date
    #     counter = 0
    #     while True:
    #         if self.hold_duration == "1d":
    #             train_end = train_start + relativedelta(months=6)
    #         elif self.hold_duration == "1w":
    #             train_end = train_start + relativedelta(years=1)
    #         else:
    #             train_end = train_start + relativedelta(years=3)
    #
    #         test_start = train_end + relativedelta(days=1)
    #         test_end = train_end + relativedelta(months=1)
    #
    #         if test_end > date.today():
    #             break
    #
    #         train = self.data[self.start_date:train_end]
    #
    #         if self.hold_duration == "1d":
    #             test = pd.concat([train.tail(5), self.data[test_start:test_end]], axis=0)
    #         elif self.hold_duration == "1w":
    #             test = pd.concat([train.tail(24), self.data[test_start:test_end]], axis=0)
    #         else:
    #             test = pd.concat([train.tail(100), self.data[test_start:test_end]], axis=0)
    #
    #         self.prepareForMC(train)
    #         self.num_weekdays = len(test)
    #
    #         monte_results = self.simulateMonteCarlo()
    #
    #         all_predictions.extend(monte_results[-1])
    #         all_tests.extend(test)
    #
    #         print(len(test))
    #         print(len(monte_results[-1]))
    #
    #         train_start += relativedelta(months=1)
    #         counter += 1
    #
    #     print(all_tests)
    #     print(all_predictions)
    #     print(len(all_tests), len(all_predictions))
