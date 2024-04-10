from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

from Regression import Regression


class MonteCarloSimulation(Regression):

    def __init__(self, num_of_simulations, prediction_date, data, num_weekdays,
                 hold_duration, start_date, is_long):
        super().__init__(hold_duration, data, prediction_date, start_date, is_long)

        self.num_of_simulations = num_of_simulations
        self.prediction_date = prediction_date
        self.num_weekdays = num_weekdays
        self.hold_duration = hold_duration
        self.start_date = start_date
        self.data = data
        self.time_increment = 1

        self.days = 1
        if self.hold_duration == "1w":
            self.days = 5
        elif self.hold_duration == "1m":
            self.days = 20

    def get_data_for_prediction(self):
        train_start = date.today() - self.historic_date_range
        data_for_prediction = self.data[train_start:]
        return data_for_prediction

    def makeMCPrediction(self, data_for_prediction):
        s_0, mu, sigma = self.prepareForMC(data_for_prediction)
        monte_results = self.simulateMonteCarlo(s_0, mu, sigma)
        return monte_results, s_0

    def displayResults(self, monte_results, s_0):
        less = 0
        more = 0
        for n in monte_results:
            m = n[-1]
            if m > s_0:
                more += 1
            elif m < s_0:
                less += 1

        percentage = (max(more, less) / self.num_of_simulations) * 100
        percentage = "{:.2f}".format(percentage)
        if more >= less:
            prediction = str(percentage) + "% chance of growth"
        else:
            prediction = str(percentage) + "% chance of fall"

        return prediction

    def prepareForMC(self, data):
        # S_0:
        s_0 = data.iloc[-1]

        # Daily Returns = R(t) = (P(t) - P(t-1)) / P(t-1) = (P(t) / P(t-1)) - 1
        # TODO: choose between daily_returns and log returns
        daily_returns = data.pct_change(self.days).dropna()
        returns = np.log(1 + daily_returns)

        # mu:
        mu = returns.mean()
        # sigma:
        sigma = returns.std()

        return s_0, mu, sigma

    def calculateTimePoints(self):
        prediction_length = int(self.num_weekdays / self.time_increment)
        time_points = np.arange(1, prediction_length + 1)
        return prediction_length, time_points

    # Method that calculates Geometric Brownian Motion:
    def calculateGBM(self, drift, prediction_length, s_0, sigma):
        # Geometric Brownian Motion formula:
        # S_t = S_0 * exp((mu - 0.5 * sigma^2) * time_points + sigma * time_points * Z)
        Z = np.random.normal(0, 1, prediction_length).cumsum()
        vol = sigma * np.sqrt(self.days) * Z
        gbm = s_0 * np.exp(drift + vol)
        return gbm

    def simulateMonteCarlo(self, s_0, mu, sigma):
        # t:
        prediction_length, time_points = self.calculateTimePoints()

        # Drift:
        drift = (mu - 0.5 * (sigma ** 2)) * self.days
        # Loop the "calculateGBM" method for Monte Carlo simulation
        monte = np.array([self.calculateGBM(drift, prediction_length, s_0, sigma)
                          for _ in range(0, self.num_of_simulations)])

        return monte
        # filtered_monte = monte[monte <= s_0 * 1.5]
        #
        # num_occurrences = len(monte) // len(filtered_monte)
        # remainder = len(monte) % len(filtered_monte)
        #
        # result = []
        #
        # for num in filtered_monte:
        #     result.extend([num] * num_occurrences)
        #
        # for i in range(remainder):
        #     result.append(monte[i])
        #
        # print(s_0)
        # print(monte)
        # print(result)

    def plotSimulation(self, monte, s_0):
        # Add starting point to each path:
        monte2 = np.hstack((np.array([[s_0] for _ in range(self.num_of_simulations)]), monte))

        today = date.today()
        x_axis = pd.date_range(start=today, end=self.prediction_date,
                               freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
        if today.weekday() >= 5:
            x_axis = pd.concat([pd.Series([today]), pd.Series(x_axis)], ignore_index=True)

        filtered_simulations = [simulation for simulation in monte2 if all(price < s_0 * 2 for price in simulation)]

        # Draw the simulation graph:
        for i in range(len(filtered_simulations)):
            plt.plot(x_axis, filtered_simulations[i], alpha=0.2)

        plt.ylabel('Price in USD')
        plt.xlabel('Prediction Days')
        plt.xticks(fontsize=9)
        plt.xticks(rotation=340)

        if self.hold_duration == "1d" or self.hold_duration == "1w":
            plt.xticks(x_axis)

        plt.axhline(y=s_0, color='r', linestyle='-')

        labels = plt.gca().get_yticklabels()

        prices = []
        for label in labels:
            try:
                prices.append(float(label.get_text()))
            except Exception:
                continue

        return prices

    def printProbabilities(self, labels, monte, s_0):
        start_price = s_0

        difference = int((labels[1] - labels[0]) / 2)
        for i in range(0, len(labels) - 1):
            labels.append(labels[i] + difference)
        labels.sort()
        prices = np.array(labels)

        less = prices[prices < start_price]
        more = prices[prices > start_price]
        test_data = [mo[-1] for mo in monte]
        growth_results = []
        fall_results = []
        num = 100 / self.num_of_simulations

        for l in less:
            percentage = round((test_data < l).sum() * num, 2)
            if percentage > 0:
                fall_results.append("< " + str(l) + ": " + str(percentage) + "%")
        for m in more:
            percentage = round((test_data > m).sum() * num, 2)
            if percentage > 0:
                growth_results.append("> " + str(m) + ": " + str(percentage) + "%")

        return growth_results, fall_results[::-1]

    def evaluateModel(self):
        # TODO: explain this sliding method clearly in report

        all_predictions = []
        all_tests = []

        train_start = date.today() - self.historic_date_range - relativedelta(days=1)

        counter = 0
        while True:
            train_end = train_start + self.historic_date_range
            dataset = self.data[train_start:train_end]

            train = dataset.iloc[:-1]
            test = dataset.iloc[-1]

            if counter == 250:
                break

            monte_results, s_0 = self.makeMCPrediction(train)
            predictions = [x[-1] for x in monte_results]
            average_prediction = sum(predictions) / len(predictions)

            all_predictions.append(average_prediction)
            all_tests.append(test)

            train_start -= relativedelta(days=1)
            counter += 1

        return super().calculateEvalMetrics(all_predictions, all_tests)
