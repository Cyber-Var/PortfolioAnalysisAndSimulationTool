from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator


class MonteCarloSimulation:

    def __init__(self, values, num_of_simulations, prediction_date, time_increment, data, weekdays, hold_duration):
        self.values = np.array(values)
        self.num_of_simulations = num_of_simulations
        self.prediction_date = prediction_date
        self.time_increment = time_increment
        self.data = data
        self.weekdays = weekdays
        self.hold_duration = hold_duration

        self.prediction_length = 0

        # S_0:
        self.s_0 = self.data.iloc[-1]
        # Daily Returns = R(t) = (P(t) - P(t-1)) / P(t-1) = (P(t) / P(t-1)) - 1
        returns = self.data.pct_change().dropna()
        # mu:
        self.mu = returns.mean()
        # sigma:
        self.sigma = returns.std()

        # Display Monte Carlo results:
        monte_results = self.simulateMonteCarlo()
        plot_labels = self.plotSimulation(monte_results)
        self.printProbabilities(plot_labels, monte_results)

    def calculateTimePoints(self):
        self.prediction_length = int(len(self.weekdays) / self.time_increment)
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

        if self.hold_duration == "1d":
            x_axis = pd.date_range(start=date.today(), end=self.prediction_date,
                                   freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        else:
            x_axis = pd.date_range(start=date.today(), end=self.prediction_date,
                                   freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()

        # Draw the simulation graph:
        for i in range(self.num_of_simulations):
            # plt.plot(monte2[i], alpha=0.5)
            plt.plot(x_axis, monte2[i], alpha=0.5)

        plt.ylabel('Price in USD')
        plt.xlabel('Prediction Days')
        plt.xticks(fontsize=9)
        plt.xticks(rotation=340)

        if self.hold_duration == "1d" or self.hold_duration == "1w":
            plt.xticks(x_axis)

        plt.axhline(y=self.s_0, color='r', linestyle='-')
        # plt.figure(figsize=(6.4, 10))

        _, labels = plt.yticks()
        plt.show()

        prices = []
        for label in labels:
            prices.append(float(label.get_text()))

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
        print(results)
