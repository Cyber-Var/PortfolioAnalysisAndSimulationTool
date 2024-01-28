from sklearn.ensemble import RandomForestRegressor

from Regression import Regression


class RandomForestRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        self.evaluateModel()
        self.makePrediction()

    def evaluateModel(self):
        Regression.reg = RandomForestRegressor()

        print("Random Forest Regression Evaluation:")
        return super().evaluateModel()

    # TODO: n_estimators chosen by user
    def makePrediction(self):
        Regression.reg = RandomForestRegressor()
        prediction = super().makePrediction()
        print("Random Forest Regression Prediction:", prediction, "\n")
        return prediction
