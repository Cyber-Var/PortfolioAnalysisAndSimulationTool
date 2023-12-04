from sklearn.ensemble import RandomForestRegressor

from Regression import Regression


class RandomForestRegressionAlgorithm(Regression):

    def __init__(self, end_date, user_end_date, data, future_data):
        super().__init__(end_date, user_end_date, data, future_data)

        X, y = super().prepareData()

        y_test, rf_predictions = self.evaluateModel(X, y)
        mse, rmse, mae, mape, r2 = super().calculateMetrics(y_test, rf_predictions)
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}%')
        print(f'R-squared: {r2}')

        rf_predictions = self.runAlgorithm(X, y)
        print("Prediction:", rf_predictions[0])

    def evaluateModel(self, X, y):
        Regression.reg = RandomForestRegressor(n_estimators=100, random_state=42)
        super().evaluateModel(X, y)
        return super().evaluateModel(X, y)

    # TODO: n_estimators chosen by user
    def runAlgorithm(self, X, y):
        Regression.reg = RandomForestRegressor(n_estimators=100, random_state=42)
        return super().runAlgorithm(X, y)
