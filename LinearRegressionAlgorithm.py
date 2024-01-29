from sklearn.linear_model import LinearRegression
from Regression import Regression


class LinearRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        self.evaluateModel()
        self.split_prediction_sets()

    def evaluateModel(self):
        Regression.reg = LinearRegression()

        print("Linear Regression Evaluation:")
        return super().evaluateModel()

    # TODO: n_estimators chosen by user
    def split_prediction_sets(self):
        Regression.reg = LinearRegression()
        X_train, y_train, X_test = super().split_prediction_sets()
        prediction = super().makePrediction(X_train, y_train, X_test)
        print("Linear Regression Prediction:", prediction, "\n")
        return prediction
