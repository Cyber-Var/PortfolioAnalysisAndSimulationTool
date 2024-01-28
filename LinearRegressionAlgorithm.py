from sklearn.linear_model import LinearRegression
from Regression import Regression


class LinearRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        self.evaluateModel()
        self.makePrediction()

    def evaluateModel(self):
        Regression.reg = LinearRegression()

        print("Linear Regression Evaluation:")
        return super().evaluateModel()

    # TODO: n_estimators chosen by user
    def makePrediction(self):
        Regression.reg = LinearRegression()
        prediction = super().makePrediction()
        print("Linear Regression Prediction:", prediction, "\n")
        return prediction
