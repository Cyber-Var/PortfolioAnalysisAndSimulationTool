from sklearn.ensemble import RandomForestRegressor

from Regression import Regression


class RandomForestRegressionAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date):
        super().__init__(hold_duration, data, prediction_date)

        X_train, y_train, X_test = super().prepareData()

        # mse, rmse, mae, mape, r2 = self.evaluateModel(X_train, y_train)
        predictions = self.runAlgorithm(X_train, y_train, X_test)

    def evaluateModel(self, X, y):
        Regression.reg = RandomForestRegressor()

        print("Random Forest Evaluation:")
        return super().evaluateModel(X, y)

    # TODO: n_estimators chosen by user
    def runAlgorithm(self, X_train, y_train, X_test):
        Regression.reg = RandomForestRegressor()
        predictions = super().runAlgorithm(X_train, y_train, X_test)
        print("Random Forest Prediction:", predictions, "\n")
        return predictions
