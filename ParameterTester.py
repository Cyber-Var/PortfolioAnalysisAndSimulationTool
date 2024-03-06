import os
import sys
from itertools import product

from ARIMAAlgorithm import ARIMAAlgorithm
from BayesianRegressionAlgorithm import BayesianRegressionAlgorithm
from LSTM import LSTMAlgorithm
from LinearRegressionAlgorithm import LinearRegressionAlgorithm
from MonteCarlo import MonteCarloSimulation
from RandomForest import RandomForestAlgorithm


class ParameterTester:

    def __init__(self, hold_duration, data, prediction_date, start_date, today_data, num_of_simulations, weekdays):
        self.hold_duration = hold_duration
        self.data = data
        self.prediction_date = prediction_date
        self.start_date = start_date
        self.today_data = today_data
        self.num_of_simulations = num_of_simulations
        self.weekdays = weekdays

        # self.tune_linear_regression_parameters()
        # self.tune_random_forest_parameters()
        # self.tune_bayesian_parameters()
        self.tune_arima_parameters()
        # self.tune_monte_carlo_parameters()

        # TODO: add multiple epochs !
        # self.tune_lstm_parameters()
        os.system('say "Your code has finished"')

    def print_best_parameters(self, parameter_grid, model_name, file_name):
        parameters_combinations = list(product(*parameter_grid.values()))

        best_mse = float('inf')
        best_mse_params = None
        best_rmse = float('inf')
        best_rmse_params = None
        best_mae = float('inf')
        best_mae_params = None
        best_mape = float('inf')
        best_mape_params = None
        best_r2 = float('-inf')
        best_r2_params = None

        counter = 1
        num = len(parameters_combinations)
        for parameters_set in parameters_combinations:
            print(f"{counter} / {num}")
            print(parameters_set)

            if model_name == "Linear Regression":
                model = LinearRegressionAlgorithm(self.hold_duration, self.data, self.prediction_date,
                                                  self.start_date, parameters_set)
            elif model_name == "Random Forest":
                model = RandomForestAlgorithm(self.hold_duration, self.data, self.prediction_date,
                                              self.start_date, parameters_set)
            elif model_name == "Bayesian":
                model = BayesianRegressionAlgorithm(self.hold_duration, self.data, self.prediction_date,
                                                    self.start_date, parameters_set)
            elif model_name == "ARIMA":
                model = ARIMAAlgorithm(self.hold_duration, self.data, self.prediction_date, self.start_date,
                                       self.today_data, parameters_set)
            elif model_name == "Monte Carlo Simulation":
                model = MonteCarloSimulation(parameters_set[0], self.prediction_date,
                                             self.data["Adj Close"], self.weekdays, self.hold_duration, self.start_date)
            else:
                model = LSTMAlgorithm(self.hold_duration, self.data, self.prediction_date, self.start_date,
                                      parameters_set)

            mse, rmse, mae, mape, r2 = model.evaluateModel()

            if mse < best_mse:
                best_mse = mse
                best_mse_params = parameters_set
            if rmse < best_rmse:
                best_rmse = rmse
                best_rmse_params = parameters_set
            if mae < best_mae:
                best_mae = mae
                best_mae_params = parameters_set
            if mape < best_mape:
                best_mape = mape
                best_mape_params = parameters_set
            if r2 > best_r2:
                best_r2 = r2
                best_r2_params = parameters_set

            counter += 1

        print(f"Best parameters for the {model_name} model:")
        print("Best MSE parameters:", best_mse_params, "      MSE =", best_mse)
        print("Best RMSE parameters:", best_rmse_params, "      RMSE =", best_rmse)
        print("Best MAE parameters:", best_mae_params, "      MAE =", best_mae)
        print("Best MAPE parameters:", best_mape_params, "      MAPE =", best_mape)
        print("Best R^2 parameters:", best_r2_params, "      R^2 =", best_r2, "\n\n")

        with open(os.path.join("parameter tuning", file_name), "a") as f:
            sys.stdout = f

            print(f"Best parameters for the {model_name} model:")
            print("Best MSE parameters:", best_mse_params, "      MSE =", best_mse)
            print("Best RMSE parameters:", best_rmse_params, "      RMSE =", best_rmse)
            print("Best MAE parameters:", best_mae_params, "      MAE =", best_mae)
            print("Best MAPE parameters:", best_mape_params, "      MAPE =", best_mape)
            print("Best R^2 parameters:", best_r2_params, "      R^2 =", best_r2, "\n\n")

            sys.stdout = sys.__stdout__

    def tune_linear_regression_parameters(self):
        parameter_grid = {
            'fit_intercept': [True, False]
        }
        self.print_best_parameters(parameter_grid, "Linear Regression", "linear_reg.txt")

    def tune_random_forest_parameters(self):
        parameter_grid = {
            'n_estimators': [50, 100],
            'max_features': ['sqrt', 'log2', 0.5, 1],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'bootstrap': [True, False],
            'criterion': ["squared_error"],
            'random_state': [42]
        }
        self.print_best_parameters(parameter_grid, "Random Forest", "random_forest.txt")

    def tune_bayesian_parameters(self):
        parameter_grid = {
            'n_iter': [100],
            'tol': [1e-3, 1e-4, 1e-5],
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-4],
            'compute_score': [True],
            'fit_intercept': [False],
            'copy_X': [True]
        }
        self.print_best_parameters(parameter_grid, "Bayesian", "bayesian.txt")

    def tune_arima_parameters(self):
        parameter_grid = {
            'maxiter': [20, 50, 100],
            'p': [0, 1, 2],
            'd': [0, 1, 2],
            'q': [0, 1, 2]
        }
        self.print_best_parameters(parameter_grid, "ARIMA", "arima.txt")

    def tune_monte_carlo_parameters(self):
        parameter_grid = {
            'num_simulations': [10, 100, 1000, 10000]
        }
        self.print_best_parameters(parameter_grid, "Monte Carlo Simulation", "monte_carlo.txt")

    def tune_lstm_parameters(self):
        parameter_grid = {
            'num_lstm_layers': [2, 3, 4, 5],
            'lstm_units': [50],
            'dropout_rate': [0, 0.2],
            'dense_units': [25],
            'optimizer': ['adam'],
            'loss': ['mean_squared_error'],
            'epochs': [10, 50]
        }
        # parameter_grid = {
        #     'num_lstm_layers': [2, 3, 4, 5],
        #     'lstm_units': [50, 100, 150],
        #     'dropout_rate': [0, 0.2, 0.3, 0.4],
        #     'dense_units': [25, 50, 75],
        #     'optimizer': ['adam', 'rmsprop'],
        #     'loss': ['mean_squared_error', 'mean_absolute_error'],
        # }
        self.print_best_parameters(parameter_grid, "LSTM", "lstm.txt")
