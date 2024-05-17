import math

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from Regression import Regression


class BayesianRegressionAlgorithm(Regression):
    """
        This class is used for initializing, training and predicting using the Bayesian Ridge regression model.
    """

    def __init__(self, hold_duration, data, prediction_date, start_date, params, is_long, investment_amount):
        """

        :param hold_duration: user selected hold duration
        :param data: data fetched for user selected the stock from API
        :param prediction_date: date for which to predict outcomes
        :param start_date: start date of the fetched historical price data
        :param params: parameters for initializing the model
        :param is_long: user selected short/long investment type
        :param investment_amount: dollar amount invested in this stock
        """
        super().__init__(hold_duration, data, prediction_date, start_date, is_long)
        self.params = params
        self.investment_amount = investment_amount

    def setup_model(self):
        """
        Method that initializes the Bayesian Ridge regression model and sets its hyperparameters
        :return: the model
        """
        model = BayesianRidge(max_iter=self.params[0], tol=self.params[1],
                              alpha_1=self.params[2], alpha_2=self.params[3],
                              lambda_1=self.params[4], lambda_2=self.params[5],
                              compute_score=self.params[6], fit_intercept=self.params[7],
                              copy_X=self.params[8])
        return model

    def evaluateModel(self):
        """
        Method that evaluates the model's performance
        :return: MAPE error metric
        """

        # Initialize model:
        Regression.reg = self.setup_model()

        # Retrieve predicted and actual prices:
        all_X_trains, all_y_trains, all_X_tests, all_y_tests = super().evaluateModel()
        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.makePrediction(all_X_trains[i], all_y_trains[i], all_X_tests[i])
            all_predictions.extend(predictions)

        # Calculate and return MAPE:
        return self.calculateEvalMetrics(all_predictions, all_y_tests)

    def predict_price(self):
        """
        Method for predicting future stock outcome
        :return: predictions
        """

        # Initialize the model:
        Regression.reg = self.setup_model()

        # Predict future price and confidence interval:
        X_train, y_train, X_test = super().split_prediction_sets()
        prediction, confidence = self.makeFinalPrediction(X_train, y_train, X_test)

        # Calculate profit/loss amount with confidence interval:
        profit_loss_amount = super().calculate_profit_or_loss(prediction, self.investment_amount)
        profit_loss_confidence = (confidence[0][0] / prediction[0][0]) * profit_loss_amount
        return profit_loss_amount, prediction, confidence, profit_loss_confidence

    def makeFinalPrediction(self, X_train, y_train, X_test):
        """
        Method for predicting future stock price
        :return: predicted price
        """

        # Scale features:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
        y_train_scaled_1d = y_train_scaled.ravel()

        X_test_scaled = scaler_X.transform(X_test)

        # Train the model:
        self.reg.fit(X_train_scaled, y_train_scaled_1d)

        # Predict future price with confidence interval:
        prediction, conf = self.reg.predict(X_test_scaled, return_std=True)
        prediction_transformed_back = scaler_y.inverse_transform(prediction.reshape(-1, 1))
        confidence_transformed_back = np.abs(scaler_y.inverse_transform(conf.reshape(-1, 1)) - prediction_transformed_back)
        return prediction_transformed_back, confidence_transformed_back
