from sklearn.ensemble import RandomForestRegressor

from Regression import Regression


class RandomForestAlgorithm(Regression):

    def __init__(self, hold_duration, data, prediction_date, start_date):
        super().__init__(hold_duration, data, prediction_date, start_date)

        self.evaluateModel()
        self.predict_price()

    def evaluateModel(self):
        Regression.reg = RandomForestRegressor()

        all_X_trains, all_X_tests, all_y_trains, all_y_tests = super().evaluateModel()

        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.makePrediction(all_X_trains[i], all_y_trains[i], all_X_tests[i])
            all_predictions.extend(predictions)

        print("Linear Regression Evaluation:")
        return self.calculateEvalMetrics(all_predictions, all_y_tests)

    # TODO: n_estimators chosen by user
    def predict_price(self):
        Regression.reg = RandomForestRegressor()
        X_train, y_train, X_test = super().split_prediction_sets()
        prediction = super().makePrediction(X_train, y_train, X_test)
        print("Random Forest Regression Prediction:", prediction, "\n")
        return prediction
