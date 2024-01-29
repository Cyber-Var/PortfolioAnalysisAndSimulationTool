from sklearn.ensemble import RandomForestRegressor

from Regression import Regression


class RandomForestRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        self.evaluateModel()
        self.split_prediction_sets()

    def evaluateModel(self):
        Regression.reg = RandomForestRegressor()

        print("Random Forest Regression Evaluation:")
        return super().evaluateModel()

    # TODO: n_estimators chosen by user
    def split_prediction_sets(self):
        Regression.reg = RandomForestRegressor()
        X_train, y_train, X_test = super().split_prediction_sets()
        prediction = super().makePrediction(X_train, y_train, X_test)
        print("Random Forest Regression Prediction:", prediction, "\n")
        return prediction
