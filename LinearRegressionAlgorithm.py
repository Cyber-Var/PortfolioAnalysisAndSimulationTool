from sklearn.linear_model import LinearRegression

from Regression import Regression


class LinearRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date, params):
        super().__init__(hold_duration, data, prediction_date, start_date)
        self.params = params

    def setup_model(self):
        model = LinearRegression(fit_intercept=self.params[0])
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
        print("Linear Regression Prediction:", prediction, "\n")
        return prediction
