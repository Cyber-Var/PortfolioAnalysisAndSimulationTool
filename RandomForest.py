from sklearn.ensemble import RandomForestRegressor

from Regression import Regression


class RandomForestAlgorithm(Regression):
    """
        This class is used for initializing, training and predicting using the Random Forest regression model.
    """

    def __init__(self, hold_duration, data, prediction_date, start_date, params, is_long, investment_amount):
        """
        Constructor method
        :param hold_duration: user selected hold duration
        :param data: data fetched for user selected the stock from API
        :param prediction_date: date for which to predict outcomes
        :param start_date: start date of the fetched historical price data
        :param params: parameters for initializing the model
        :param is_long: user selected short/long investment type
        :param investment_amount: dollar amount invested in this stock
        """
        super().__init__(hold_duration, data, prediction_date, start_date, is_long)
        self.params = params
        self.investment_amount = investment_amount

    def setup_model(self):
        """
        Method that initializes the Random forest regression model and sets its hyperparameters
        :return: the model
        """
        model = RandomForestRegressor(n_estimators=self.params[0], max_features=self.params[1],
                                               max_depth=self.params[2], min_samples_split=self.params[3],
                                               min_samples_leaf=self.params[4], bootstrap=self.params[5],
                                               criterion=self.params[6], random_state=self.params[7])
        return model

    def evaluateModel(self):
        """
        Method that evaluates the model's performance
        :return: MAPE error metric
        """

        # Initialize model:
        Regression.reg = self.setup_model()

        # Retrieve predicted and actual prices:
        all_X_trains, all_y_trains, all_X_tests, all_y_tests = super().evaluateModel()
        all_predictions = []
        for i in range(len(all_X_trains)):
            predictions = self.makePrediction(all_X_trains[i], all_y_trains[i], all_X_tests[i])
            all_predictions.extend(predictions)

        # Calculate and return MAPE:
        return self.calculateEvalMetrics(all_predictions, all_y_tests)

    def predict_price(self):
        """
                Method for predicting future stock outcome
                :return: predictions
                """

        # Initialize the model:
        Regression.reg = self.setup_model()

        # Predict future price:
        X_train, y_train, X_test = super().split_prediction_sets()
        prediction = super().makePrediction(X_train, y_train, X_test)

        # Calculate profit/loss amount:
        profit_loss_amount = super().calculate_profit_or_loss(prediction, self.investment_amount)
        return profit_loss_amount, prediction
