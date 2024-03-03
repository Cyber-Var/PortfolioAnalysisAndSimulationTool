from sklearn.linear_model import BayesianRidge

from Regression import Regression


class BayesianRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date, params):
        super().__init__(hold_duration, data, prediction_date, start_date)
        self.params = params

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
        prediction = super().makePrediction(X_train, y_train, X_test)
        return prediction
