from datetime import date
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


class Regression:
    """
    Helper class for each algorithm from the suite
    """
    reg = None

    def __init__(self, hold_duration, data, prediction_date, start_date, is_long):
        """
        Constructor method
        :param hold_duration: user selected hold duration
        :param data: historical data for analysis
        :param prediction_date: date for which to make predictions
        :param start_date: starting historical date
        :param is_long: long/short investment
        """
        self.hold_duration = hold_duration
        self.data = data
        self.prediction_date = prediction_date
        self.start_date = start_date
        self.is_long = is_long

        if self.hold_duration == "1d":
            self.days = 5
            self.num_days = 0
            self.historic_date_range = relativedelta(months=6)
        elif self.hold_duration == "1w":
            self.days = 20
            self.num_days = 4
            self.historic_date_range = relativedelta(years=1)
        else:
            self.days = 80
            self.num_days = 20
            self.historic_date_range = relativedelta(years=3)

    def evaluateModel(self):
        """
        Function for evaluating a model's performance
        :return: MAPE error metric
        """

        all_X_trains = []
        all_X_tests = []
        all_y_trains = []
        all_y_tests = []

        # Historical data start date:
        train_start = self.start_date
        counter = 0
        while True:
            train_end, test_start, test_end = self.get_train_test_dates(train_start)

            if test_end > date.today():
                break

            # Predict prices:
            train, test = self.split_train_test_sets(train_end, test_start, test_end)
            X_train, y_train, X_test, y_test = self.prepareData(train, test, True)

            # Store predicted and actual prices:
            all_X_trains.append(X_train)
            all_X_tests.append(X_test)
            all_y_trains.append(y_train)
            all_y_tests.extend(y_test)

            train_start += relativedelta(months=1)
            counter += 1

        # Return the sets:
        return all_X_trains, all_y_trains, all_X_tests, all_y_tests

    def get_train_test_dates(self, train_start):
        """
        Get the dates for training and testing:
        :param train_start: start date for historical analysis
        :return: training end date, testing start date, testing end date
        """
        train_end = train_start + self.historic_date_range
        test_start = train_end + relativedelta(days=1)
        test_end = train_end + relativedelta(months=1)
        return train_end, test_start, test_end

    def split_train_test_sets(self, train_end, test_start, test_end):
        """
        Split the data into training and testing set:
        :param train_end: training end date
        :param test_start: testing start date
        :param test_end: testing end date
        :return: train set and test set
        """
        train = self.data[self.start_date:train_end]
        test = pd.concat([train.tail(self.days + self.num_days), self.data[test_start:test_end]], axis=0)
        return train, test

    def split_prediction_sets(self):
        """
        Split the historical data into X_train, y_train, X_test
        :return: X_train, y_train, X_test
        """
        if self.hold_duration == "1d":
            train_start = date.today() - relativedelta(months=6)
        elif self.hold_duration == "1w":
            train_start = date.today() - relativedelta(years=1)
        else:
            train_start = date.today() - relativedelta(years=3)
        train = self.data[train_start:]

        X_train, y_train, X_test, y_test = self.prepareData(train, [], False)
        return X_train, y_train, X_test

    def makePrediction(self, X_train, y_train, X_test):
        """
        Predict future price
        :param X_train: training input features
        :param y_train: training target values
        :param X_test: testing input features
        :return: predicted price
        """

        # Scale the features:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
        y_train_scaled_1d = y_train_scaled.ravel()
        X_test_scaled = scaler_X.transform(X_test)

        # Train the model:
        self.reg.fit(X_train_scaled, y_train_scaled_1d)

        # Predict price:
        prediction = self.reg.predict(X_test_scaled).reshape(-1, 1)
        return scaler_y.inverse_transform(prediction)

    def prepareData(self, train_set, test_set, evaluate):
        """
        Method for preparing data for analysis:
        :param train_set: training set
        :param test_set: testing set
        :param evaluate: True if evaluating the model, False if simply predicting
        :return:
        """
        X_train, y_train = self.split_X_y(train_set)

        if evaluate:
            X_test, y_test = self.split_X_y(test_set)
        else:
            X_test = [self.process_features(train_set[-self.days:])]
            y_test = []

        return X_train, y_train, X_test, y_test

    def split_X_y(self, data):
        """
        Method for splitting a set into input features and target values
        :param data: data to split
        :return: input features and target values
        """
        X = [self.process_features(data[i:i + self.days]) for i in range(0, len(data) - (self.days + self.num_days))]
        y = data["Adj Close"].iloc[(self.days + self.num_days):].values.tolist()
        return X, y

    def process_features(self, li):
        """
        Method that performs feature engineering
        :param li: dataset
        :return: feature engineered dataset
        """

        # Add Adj Close prices to features:
        if self.hold_duration == "1d":
            result = li['Adj Close'].values.tolist()
            # Daily changes (differences):
            changes = li["Adj Close"].diff().dropna()
        elif self.hold_duration == "1w":
            result = np.convolve(li['Adj Close'], np.ones(7) / 7, mode='valid').tolist()
            # Weekly changes (differences):
            changes = li['Adj Close'].rolling(window=6).apply(
                lambda x: (x.iloc[0] - x.iloc[5]))
        else:
            result = np.convolve(li['Adj Close'], np.ones(60) / 60, mode='valid').tolist()
            # Monthly changes (differences):
            changes = li['Adj Close'].rolling(window=21).apply(
                lambda x: (x.iloc[0] - x.iloc[20]))

        # Add Moving Averages to features:
        result += [changes.mean()]

        # Add opening, high, low prices and volumes to features:
        result += [li['Adj Close'].mean(), li['Open'].mean(), li['Close'].mean(), li['High'].mean(), li['Low'].mean(),
                   li['Volume'].mean()]
        return result

    def calculateEvalMetrics(self, predictions, y_test):
        """
        Method that calculates MAPE score
        :param predictions: predicted prices
        :param y_test: actual prices
        :return: MAPE
        """
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        return mape

    def calculate_profit_or_loss(self, predicted_price, investment_amount):
        """
        Method that calculates profit/loss amount from predicted price
        :param predicted_price: predicted price
        :param investment_amount: amount invested in stock
        :return: profit/loss amount
        """
        current_price = self.data.iloc[-1]["Adj Close"]
        percentage_change = (predicted_price - current_price) / current_price
        profit_or_loss_value = investment_amount * percentage_change
        if not self.is_long:
            profit_or_loss_value = -profit_or_loss_value
        return profit_or_loss_value
