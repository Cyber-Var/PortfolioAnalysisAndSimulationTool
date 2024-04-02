import math

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from Regression import Regression


class BayesianRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date, params, is_long, investment_amount):
        super().__init__(hold_duration, data, prediction_date, start_date, is_long)
        self.params = params
        self.investment_amount = investment_amount

    def setup_model(self):
        model = BayesianRidge(max_iter=self.params[0], tol=self.params[1],
                              alpha_1=self.params[2], alpha_2=self.params[3],
                              lambda_1=self.params[4], lambda_2=self.params[5],
                              compute_score=self.params[6], fit_intercept=self.params[7],
                              copy_X=self.params[8])
        return model

    def evaluateModel(self):
        Regression.reg = self.setup_model()

        all_X_trains, all_y_trains, all_X_tests, all_y_tests = super().evaluateModel()

        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.makePrediction(all_X_trains[i], all_y_trains[i], all_X_tests[i])
            all_predictions.extend(predictions)

        return self.calculateEvalMetrics(all_predictions, all_y_tests)

    # TODO: n_estimators chosen by user
    def predict_price(self):
        Regression.reg = self.setup_model()
        X_train, y_train, X_test = super().split_prediction_sets()
        prediction, confidence = self.makeFinalPrediction(X_train, y_train, X_test)
        profit_loss_amount = super().calculate_profit_or_loss(prediction, self.investment_amount)
        print(f"Predicted price: {prediction[0][0]} +/- {confidence[0][0]} pounds.")

        profit_loss_confidence = (confidence[0][0] / prediction[0][0]) * profit_loss_amount
        print(f"Predicted profit/loss: {profit_loss_amount} +/- {profit_loss_confidence} pounds.")
        return profit_loss_amount, prediction, confidence, profit_loss_confidence

    def makeFinalPrediction(self, X_train, y_train, X_test):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
        y_train_scaled_1d = y_train_scaled.ravel()

        X_test_scaled = scaler_X.transform(X_test)
        self.reg.fit(X_train_scaled, y_train_scaled_1d)
        # prediction = self.reg.predict(X_test_scaled).reshape(-1, 1)
        prediction, conf = self.reg.predict(X_test_scaled, return_std=True)
        prediction_transformed_back = scaler_y.inverse_transform(prediction.reshape(-1, 1))
        confidence_transformed_back = np.abs(scaler_y.inverse_transform(conf.reshape(-1, 1)) - prediction_transformed_back)
        return prediction_transformed_back, confidence_transformed_back
