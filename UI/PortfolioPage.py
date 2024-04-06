import re

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox,
                             QScrollArea, QDialog, QLineEdit, QCompleter, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel
import yfinance as yf

from UI.Page import Page
from UI.SingleStockPage import SingleStockPage


class PortfolioPage(QWidget, Page):

    back_to_menu_page = pyqtSignal()
    open_single_stock_page = pyqtSignal(str, str, float, int, str)

    def __init__(self, main_window, controller, set_algorithms, hold_duration):
        super().__init__()

        self.main_window = main_window
        self.controller = controller
        self.set_algorithms = set_algorithms
        self.tickers = []

        self.hold_duration = hold_duration
        self.hold_duration_label = None
        self.hold_duration_1d = None
        self.hold_duration_1w = None
        self.hold_duration_1m = None

        self.algorithms_label = None
        self.algorithm_1 = None
        self.algorithm_2 = None
        self.algorithm_3 = None
        self.algorithm_4 = None
        self.algorithm_5 = None
        self.algorithm_6 = None
        self.algorithms = [False] * 6

        self.outer_ranking_vbox = QVBoxLayout()
        self.outer_ranking_vbox.setContentsMargins(0, 70, 0, 0)

        self.ranking_vbox = QVBoxLayout()
        self.ranking_widget = QWidget()
        self.ranking_widget.setObjectName("rankingHBox")
        self.ranking_widget.setFixedSize(200, 250)
        self.ranking_widget.setLayout(self.ranking_vbox)

        self.outer_ranking_vbox.addWidget(self.ranking_widget)

        self.rankings = self.controller.handle_ranking()
        ranking_label = QLabel("Ranking:")
        ranking_label.setObjectName("inputHeaderLabel")
        ranking_label.setFixedWidth(150)
        ranking_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ranking_vbox.addWidget(ranking_label)

        for algorithm in self.rankings["1d"]:
            ranking_result_label = QLabel()
            ranking_result_label.setObjectName("inputLabel")
            ranking_result_label.setFixedWidth(150)
            ranking_result_label.setAlignment(QtCore.Qt.AlignCenter)
            self.ranking_vbox.addWidget(ranking_result_label)

        update_ranking_button = QPushButton("Update Ranking")
        update_ranking_button.setObjectName('portfolioButton')
        update_ranking_button.setFixedWidth(150)
        update_ranking_button.clicked.connect(self.show_ranking_time_warning_window)
        self.ranking_vbox.addWidget(update_ranking_button, alignment=Qt.AlignCenter)

        self.portfolio_results = []
        self.portfolio_amount = None
        self.portfolio_linear_regression = None
        self.portfolio_random_forest = None
        self.portfolio_bayesian = None
        self.portfolio_monte_carlo = None
        self.portfolio_lstm = None
        self.portfolio_arima = None
        self.portfolio_volatility = None
        self.portfolio_sharpe_ratio = None
        self.portfolio_VaR = None

        self.result_col_names = []
        self.ticker_col_name = None
        self.stock_name_col_name = None
        self.amount_col_name = None
        self.lin_reg_col_name = None
        self.random_forest_col_name = None
        self.bayesian_col_name = None
        self.monte_carlo_col_name = None
        self.lstm_col_name = None
        self.arima_col_name = None
        self.volatility_col_name = None
        self.sharpe_ratio_col_name = None
        self.VaR_col_name = None
        self.more_info_col_name = None
        self.edit_col_name = None

        self.results_vbox = None
        self.results_map = {}

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Portfolio Analysis")

        self.build_page()

        self.setStyleSheet(self.load_stylesheet())

        self.setLayout(self.layout)

    def build_page(self):
        self.logger.info('Building the Portfolio Page')

        title_label = self.get_title_label("Portfolio Analysis and Simulation Tool", "titleLabelPortfolio")

        self.hold_duration_label = QLabel("Hold Duration:")
        self.hold_duration_label.setObjectName('inputHeaderLabel')

        hold_duration_vbox = QVBoxLayout()

        hold_duration_hbox = QHBoxLayout()
        hold_duration_widget = QWidget()
        hold_duration_widget.setObjectName("inputHBox")
        hold_duration_widget.setFixedSize(300, 200)
        hold_duration_widget.setLayout(hold_duration_hbox)

        self.hold_duration_1d = self.create_hold_duration_button("1 day")
        self.hold_duration_1w = self.create_hold_duration_button("1 week")
        self.hold_duration_1m = self.create_hold_duration_button("1 month")

        if self.hold_duration == "1w":
            self.hold_duration_1w.setChecked(True)
        elif self.hold_duration == "1m":
            self.hold_duration_1m.setChecked(True)
        else:
            self.hold_duration_1d.setChecked(True)

        hold_duration_vbox.addWidget(self.hold_duration_1d)
        hold_duration_vbox.addWidget(self.hold_duration_1w)
        hold_duration_vbox.addWidget(self.hold_duration_1m)

        hold_duration_hbox.addWidget(self.hold_duration_label)
        hold_duration_hbox.addLayout(hold_duration_vbox)

        self.results_vbox = QVBoxLayout()

        column_names_hbox = QHBoxLayout()
        column_names_hbox.setSpacing(3)

        self.ticker_col_name = self.create_column_names_labels("Ticker")
        self.ticker_col_name.setFixedSize(60, 50)
        self.stock_name_col_name = self.create_column_names_labels("Name")
        self.stock_name_col_name.setFixedSize(160, 50)
        self.amount_col_name = self.create_column_names_labels("Investment\nAmount")
        self.amount_col_name.setFixedSize(105, 50)
        self.lin_reg_col_name = self.create_column_names_labels("Linear\nRegression")
        self.lin_reg_col_name.setFixedSize(85, 50)
        self.lin_reg_col_name.hide()
        self.random_forest_col_name = self.create_column_names_labels("Random\nForest")
        self.random_forest_col_name.setFixedSize(80, 50)
        self.random_forest_col_name.hide()
        self.bayesian_col_name = self.create_column_names_labels("Bayesian")
        self.bayesian_col_name.setFixedSize(120, 50)
        self.bayesian_col_name.hide()
        self.monte_carlo_col_name = self.create_column_names_labels("Monte Carlo")
        self.monte_carlo_col_name.setFixedSize(210, 50)
        self.monte_carlo_col_name.hide()
        self.lstm_col_name = self.create_column_names_labels("LSTM")
        self.lstm_col_name.setFixedSize(80, 50)
        self.lstm_col_name.hide()
        self.arima_col_name = self.create_column_names_labels("ARIMA")
        self.arima_col_name.setFixedSize(120, 50)
        self.arima_col_name.hide()
        self.volatility_col_name = self.create_column_names_labels("Volatility")
        self.volatility_col_name.setFixedSize(80, 50)
        self.sharpe_ratio_col_name = self.create_column_names_labels("Sharpe\nRatio")
        self.sharpe_ratio_col_name.setFixedSize(80, 50)
        self.VaR_col_name = self.create_column_names_labels("Value\nat Risk")
        self.VaR_col_name.setFixedSize(60, 50)
        self.more_info_col_name = self.create_column_names_labels("More\nInfo")
        self.more_info_col_name.setFixedSize(50, 50)
        self.edit_col_name = self.create_column_names_labels("Edit\nStock")
        self.edit_col_name.setFixedSize(50, 50)
        self.result_col_names = [self.lin_reg_col_name, self.random_forest_col_name, self.bayesian_col_name,
                                 self.monte_carlo_col_name, self.lstm_col_name, self.arima_col_name]

        column_names_hbox.addWidget(self.ticker_col_name)
        column_names_hbox.addWidget(self.stock_name_col_name)
        column_names_hbox.addWidget(self.amount_col_name)
        column_names_hbox.addWidget(self.lin_reg_col_name)
        column_names_hbox.addWidget(self.random_forest_col_name)
        column_names_hbox.addWidget(self.bayesian_col_name)
        column_names_hbox.addWidget(self.monte_carlo_col_name)
        column_names_hbox.addWidget(self.lstm_col_name)
        column_names_hbox.addWidget(self.arima_col_name)
        column_names_hbox.addWidget(self.volatility_col_name)
        column_names_hbox.addWidget(self.sharpe_ratio_col_name)
        column_names_hbox.addWidget(self.VaR_col_name)
        column_names_hbox.addWidget(self.more_info_col_name)
        column_names_hbox.addWidget(self.edit_col_name)

        add_stock_button = QPushButton("+ Add Stock")
        add_stock_button.setObjectName('addStockButton')
        add_stock_button.clicked.connect(self.show_add_stock_window)

        column_names_hbox.addStretch(1)

        scrollable_area = QScrollArea()
        scrollable_area.setWidgetResizable(True)
        scrollable_widget = QWidget()
        scrollable_widget.setStyleSheet("border: 2px solid white;")
        scrollable_layout = QVBoxLayout()
        scrollable_layout.addLayout(column_names_hbox)
        scrollable_layout.addLayout(self.results_vbox)
        scrollable_layout.addWidget(add_stock_button)
        scrollable_layout.addStretch(1)
        scrollable_widget.setLayout(scrollable_layout)
        scrollable_area.setWidget(scrollable_widget)

        back_button = QPushButton("Back")
        back_button.setObjectName("addStockButton")
        back_button.clicked.connect(self.back_to_menu_page.emit)

        self.show_portfolio_results()

        for ticker in self.controller.tickers_and_investments.keys():
            data = yf.Ticker(ticker)
            one_day_data = data.history(period="1d")
            one_share_price = round(one_day_data["Close"].iloc[0], 2)
            self.add_ticker(ticker, one_share_price, self.controller.tickers_and_num_shares[ticker],
                            self.controller.tickers_and_long_or_short[ticker])

        algrithms_hbox = QHBoxLayout()
        algorithms_widget = QWidget()
        algorithms_widget.setObjectName("inputHBox")
        algorithms_widget.setFixedSize(300, 200)
        algorithms_widget.setLayout(algrithms_hbox)

        self.algorithms_label = QLabel("Algorithms:")
        self.algorithms_label.setObjectName('inputHeaderLabel')

        algorithms_vbox = QVBoxLayout()

        self.algorithm_1 = self.create_algorithm_checkbox("Linear Regression", 0)
        self.algorithm_2 = self.create_algorithm_checkbox("Random Forest", 1)
        self.algorithm_3 = self.create_algorithm_checkbox("Bayesian", 2)
        self.algorithm_4 = self.create_algorithm_checkbox("Monte Carlo", 3)
        self.algorithm_5 = self.create_algorithm_checkbox("LSTM", 4)
        self.algorithm_6 = self.create_algorithm_checkbox("ARIMA", 5)

        algorithms_vbox.addWidget(self.algorithm_1)
        algorithms_vbox.addWidget(self.algorithm_2)
        algorithms_vbox.addWidget(self.algorithm_3)
        algorithms_vbox.addWidget(self.algorithm_4)
        algorithms_vbox.addWidget(self.algorithm_5)
        algorithms_vbox.addWidget(self.algorithm_6)

        algrithms_hbox.addWidget(self.algorithms_label)
        algrithms_hbox.addLayout(algorithms_vbox)

        input_hbox = QHBoxLayout()
        input_hbox.setSpacing(150)
        input_hbox.addWidget(hold_duration_widget)
        input_hbox.addWidget(algorithms_widget)

        top_hbox = QHBoxLayout()

        top_left_vbox = QVBoxLayout()
        title_label.setAlignment(Qt.AlignRight)
        top_left_vbox.addWidget(title_label)
        top_left_vbox.addLayout(input_hbox)

        top_hbox.addLayout(top_left_vbox)
        top_hbox.addLayout(self.outer_ranking_vbox)

        self.layout.addLayout(top_hbox)
        self.layout.addWidget(scrollable_area)
        self.layout.addWidget(back_button)

    def show_portfolio_results(self):
        portfolio_hbox = QHBoxLayout()
        portfolio_hbox.setSpacing(3)

        portfolio_label = QLabel("Overall Portfolio results")
        portfolio_label.setObjectName("resultLabel")
        portfolio_label.setFixedSize(223, 70)

        self.portfolio_amount = QLabel("-")
        self.portfolio_amount.setObjectName("resultLabel")
        self.portfolio_amount.setFixedSize(105, 70)

        self.portfolio_linear_regression = QLabel("")
        self.portfolio_linear_regression.setFixedSize(85, 70)
        self.portfolio_random_forest = QLabel("")
        self.portfolio_random_forest.setFixedSize(80, 70)
        self.portfolio_bayesian = QLabel("")
        self.portfolio_bayesian.setFixedSize(120, 70)
        self.portfolio_monte_carlo = QLabel("")
        self.portfolio_monte_carlo.setFixedSize(210, 70)
        self.portfolio_lstm = QLabel("")
        self.portfolio_lstm.setFixedSize(80, 70)
        self.portfolio_arima = QLabel("")
        self.portfolio_arima.setFixedSize(120, 70)

        self.portfolio_linear_regression.setObjectName("resultLabel")
        self.portfolio_random_forest.setObjectName("resultLabel")
        self.portfolio_bayesian.setObjectName("resultLabel")
        self.portfolio_monte_carlo.setObjectName("resultLabel")
        self.portfolio_lstm.setObjectName("resultLabel")
        self.portfolio_arima.setObjectName("resultLabel")

        self.portfolio_linear_regression.setFixedHeight(70)
        self.portfolio_random_forest.setFixedHeight(70)
        self.portfolio_bayesian.setFixedHeight(70)
        self.portfolio_monte_carlo.setFixedHeight(70)
        self.portfolio_lstm.setFixedHeight(70)
        self.portfolio_arima.setFixedHeight(70)

        self.portfolio_linear_regression.hide()
        self.portfolio_random_forest.hide()
        self.portfolio_bayesian.hide()
        self.portfolio_monte_carlo.hide()
        self.portfolio_lstm.hide()
        self.portfolio_arima.hide()

        self.portfolio_results = [self.portfolio_linear_regression, self.portfolio_random_forest,
                                  self.portfolio_bayesian, self.portfolio_monte_carlo, self.portfolio_lstm,
                                  self.portfolio_arima]

        self.portfolio_volatility = QLabel("")
        self.portfolio_volatility.setObjectName("resultLabel")
        self.portfolio_volatility.setFixedSize(80, 70)

        self.portfolio_sharpe_ratio = QLabel("")
        self.portfolio_sharpe_ratio.setObjectName("resultLabel")
        self.portfolio_sharpe_ratio.setFixedSize(80, 70)

        self.portfolio_VaR = QLabel("")
        self.portfolio_VaR.setObjectName("resultLabel")
        self.portfolio_VaR.setFixedSize(60, 70)

        portfolio_more_info_label = QLabel("-")
        portfolio_more_info_label.setObjectName("resultLabel")
        portfolio_more_info_label.setFixedSize(50, 70)

        portfolio_edit_label = QLabel("-")
        portfolio_edit_label.setObjectName("resultLabel")
        portfolio_edit_label.setFixedSize(50, 70)

        portfolio_hbox.addWidget(portfolio_label)
        portfolio_hbox.addWidget(self.portfolio_amount)
        portfolio_hbox.addWidget(self.portfolio_linear_regression)
        portfolio_hbox.addWidget(self.portfolio_random_forest)
        portfolio_hbox.addWidget(self.portfolio_bayesian)
        portfolio_hbox.addWidget(self.portfolio_monte_carlo)
        portfolio_hbox.addWidget(self.portfolio_lstm)
        portfolio_hbox.addWidget(self.portfolio_arima)
        portfolio_hbox.addWidget(self.portfolio_volatility)
        portfolio_hbox.addWidget(self.portfolio_sharpe_ratio)
        portfolio_hbox.addWidget(self.portfolio_VaR)
        portfolio_hbox.addWidget(portfolio_more_info_label)
        portfolio_hbox.addWidget(portfolio_edit_label)

        portfolio_hbox.addStretch(1)
        self.results_vbox.addLayout(portfolio_hbox)

    def create_hold_duration_button(self, name):
        button = QRadioButton(name)
        button.setObjectName('inputLabel')
        button.toggled.connect(self.hold_duration_button_toggled)
        return button

    def create_algorithm_checkbox(self, name, ii):
        button = QCheckBox(name)
        button.setObjectName('inputLabel')
        button.stateChanged.connect(lambda state, index=ii: self.algorithms_state_changed(state, index))
        if self.set_algorithms[ii]:
            button.setChecked(True)
        return button

    def create_column_names_labels(self, name):
        label = QLabel(name)
        label.setObjectName('columnNameLabel')
        return label

    def hold_duration_button_toggled(self, checked):
        if checked:
            self.logger.info('Handling the change in hold duration.')
            if self.hold_duration_1d.isChecked():
                self.hold_duration = "1d"
                self.update_ranking_display()
                self.update_shares_results()
            elif self.hold_duration_1w.isChecked():
                self.hold_duration = "1w"
                self.update_ranking_display()
                self.update_shares_results()
            elif self.hold_duration_1m.isChecked():
                self.hold_duration = "1m"
                self.update_ranking_display()
                self.update_shares_results()

    def update_ranking_display(self):
        for alg_index, algorithm in enumerate(self.rankings[self.hold_duration]):
            ranking_label = QLabel(algorithm)
            ranking_label.setObjectName('rankingLabel')
            self.ranking_vbox.itemAt(alg_index + 1).widget().setText(f"{alg_index + 1}. {algorithm}")

    def result_to_string(self, result):
        if result > 0:
            return f"+{result:.2f}$"
        return f"{result:.2f}$"

    def algorithms_state_changed(self, state, index):
        if state == Qt.Checked:
            self.algorithms[index] = True
            self.update_algorithm_values(index)
        else:
            self.algorithms[index] = False
            for ticker in self.controller.tickers_and_investments.keys():
                self.results_map[ticker].itemAt(3 + index).widget().hide()
                self.result_col_names[index].hide()
                self.update_portfolio_results()

    def update_algorithm_values(self, index):
        self.logger.info('Updating the Algorithmic results')

        algorithm_name = self.controller.algorithms_with_indices[index]
        algorithmic_results = self.controller.results[algorithm_name][self.hold_duration]
        for ticker in self.controller.tickers_and_investments.keys():
            label = self.results_map[ticker].itemAt(3 + index).widget()
            if ticker not in algorithmic_results.keys():
                self.controller.run_algorithm(ticker, index, self.hold_duration)
            if index == 2:
                label.setText(f"{self.result_to_string(algorithmic_results[ticker])} +/- "
                              f"{self.controller.bayesian_confidences[self.hold_duration][ticker][1]:.2f}")
            elif index == 3:
                res = algorithmic_results[ticker]
                growth_fall = res.split()
                is_long = self.controller.tickers_and_long_or_short[ticker]
                if (growth_fall == "growth" and is_long) or (growth_fall == "fall" and not is_long):
                    res += " (profit)"
                else:
                    res += " (loss)"
                label.setText(res)
            elif index == 5:
                label.setText(f"{self.result_to_string(algorithmic_results[ticker])} +/- "
                              f"{self.controller.arima_confidences[self.hold_duration][ticker][1]:.2f}")
            else:
                result = algorithmic_results[ticker]
                label.setText(self.result_to_string(result))
            label.show()
            self.result_col_names[index].show()
            self.update_portfolio_results()

    def update_portfolio_results(self):
        self.logger.info('Updating the Portfolio results')

        if len(self.controller.tickers_and_investments) == 0:
            self.portfolio_amount.setText("-")
            self.portfolio_volatility.setText("-")
            self.portfolio_sharpe_ratio.setText("-")
            self.portfolio_VaR.setText("-")

            for col_name_label in self.result_col_names:
                col_name_label.hide()

            for index, is_chosen in enumerate(self.algorithms):
                if is_chosen:
                    self.portfolio_results[index].hide()
            return

        self.portfolio_amount.setText(str(sum(self.controller.tickers_and_investments.values())) + "$")
        for index, is_chosen in enumerate(self.algorithms):
            if is_chosen:
                if index == 3:
                    result = self.controller.calculate_portfolio_monte_carlo(self.hold_duration)
                else:
                    num_result = self.controller.calculate_portfolio_result(index, self.hold_duration)
                    result = self.result_to_string(num_result)
                self.portfolio_results[index].setText(result)
                self.portfolio_results[index].show()
            else:
                self.portfolio_results[index].hide()

        ticker_keys = self.controller.tickers_and_investments.keys()
        if len(ticker_keys) == 1:
            ticker = list(ticker_keys)[0]
            portfolio_vol, portfolio_vol_cat = self.controller.volatilities[ticker]
            self.portfolio_volatility.setText(f'{portfolio_vol:.2f} {portfolio_vol_cat}')

            portfolio_sharpe, portfolio_share_cat = self.controller.sharpe_ratios[ticker]
            self.portfolio_sharpe_ratio.setText(f'{portfolio_sharpe:.2f} {portfolio_share_cat}')

            self.portfolio_VaR.setText(f"{self.controller.VaRs[ticker]:.2f}")
            return

        portfolio_vol, portfolio_vol_cat = self.controller.get_portfolio_volatility(self.hold_duration)
        self.portfolio_volatility.setText(f'{portfolio_vol:.2f} {portfolio_vol_cat}')

        portfolio_sharpe, portfolio_share_cat = self.controller.get_portfolio_sharpe_ratio(self.hold_duration, portfolio_vol)
        self.portfolio_sharpe_ratio.setText(f'{portfolio_sharpe:.2f} {portfolio_share_cat}')

        self.portfolio_VaR.setText(str(self.controller.get_portfolio_VaR(self.hold_duration, portfolio_vol)))

    def update_shares_results(self):
        for index, is_chosen in enumerate(self.algorithms):
            if is_chosen:
                self.update_algorithm_values(index)

    def show_add_stock_window(self):
        popup = AddStockPopUp(self.controller.get_sp500_tickers(), self.controller.get_top_50_esg_companies())
        popup.valid_ticker_entered.connect(self.add_ticker)
        popup.exec_()

    def show_ranking_time_warning_window(self):
        popup = RankingTimeWarningPopUp()
        popup.decision_made.connect(self.process_ranking_request)
        popup.exec_()

    def process_ranking_request(self, should_process):
        if should_process:
            self.rankings = self.controller.handle_ranking(True)
            self.update_ranking_display()

    widths = [60, 160, 105, 85, 80, 120, 210, 80, 120, 80, 80, 60]

    def add_ticker(self, ticker, one_share_price, num_shares, is_long, not_initial=True):
        self.logger.info('Adding new stock to portfolio.')

        investment = round(one_share_price * num_shares, 2)

        if not_initial:
            self.controller.add_ticker(ticker, num_shares, investment, is_long)
        # self.controller.tickers_and_num_shares[ticker] = num_shares

        results_hbox = QHBoxLayout()
        results_hbox.setSpacing(3)

        for i in range(12):
            label = QLabel()
            label.setObjectName("resultLabel")
            label.setFixedSize(self.widths[i], 50)
            if 3 <= i <= 8:
                label.hide()
            results_hbox.addWidget(label)

        more_info_button = QPushButton("--->")
        more_info_button.setObjectName("moreInfoButton")
        more_info_button.setFixedSize(50, 50)
        more_info_button.clicked.connect(lambda: self.open_single_stock_page.emit(ticker, stock_name, one_share_price,
                                                                                  num_shares, self.hold_duration))
        results_hbox.addWidget(more_info_button)

        edit_button = QPushButton("Edit")
        edit_button.setObjectName("moreInfoButton")
        edit_button.setFixedSize(50, 50)
        edit_button.clicked.connect(lambda: self.edit_stock(ticker, stock_name, one_share_price))
        results_hbox.addWidget(edit_button)

        results_hbox.itemAt(0).widget().setText(ticker)

        stock_info = yf.Ticker(ticker).info
        stock_name = stock_info.get('longName', 'N/A')
        results_hbox.itemAt(1).widget().setText(stock_name)

        if is_long:
            long_short_str = "Long"
        else:
            long_short_str = "Short"
        results_hbox.itemAt(2).widget().setText("$" + str(investment) + " " + long_short_str)

        if self.algorithms[0]:
            self.lin_reg_col_name.show()
            lin_reg_prediction = self.controller.run_linear_regression(ticker, self.hold_duration)
            results_hbox.itemAt(3).widget().setText(self.result_to_string(lin_reg_prediction))
            results_hbox.itemAt(3).widget().show()
        if self.algorithms[1]:
            self.random_forest_col_name.show()
            random_forest_prediction = self.controller.run_random_forest(ticker, self.hold_duration)
            results_hbox.itemAt(4).widget().setText(self.result_to_string(random_forest_prediction))
            results_hbox.itemAt(4).widget().show()
        if self.algorithms[2]:
            self.bayesian_col_name.show()
            bayesian_prediction = self.controller.run_bayesian(ticker, self.hold_duration)
            results_hbox.itemAt(5).widget().setText(f"{self.result_to_string(bayesian_prediction[0][0])}"
                                                    f" +/- {self.controller.bayesian_confidences[self.hold_duration][ticker][1]:.2f}")
            results_hbox.itemAt(5).widget().show()
        if self.algorithms[3]:
            self.monte_carlo_col_name.show()
            # TODO: num_of_simulations set by user ??
            monte_carlo_prediction = self.controller.run_monte_carlo(ticker, self.hold_duration)
            growth_fall = monte_carlo_prediction.split()
            if (growth_fall == "growth" and is_long) or (growth_fall == "fall" and not is_long):
                monte_carlo_prediction += " (profit)"
            else:
                monte_carlo_prediction += " (loss)"
            results_hbox.itemAt(6).widget().setText(monte_carlo_prediction)
            results_hbox.itemAt(6).widget().show()
        if self.algorithms[4]:
            self.lstm_col_name.show()
            lstm_prediction = self.controller.run_lstm(ticker, self.hold_duration)
            results_hbox.itemAt(7).widget().setText(self.result_to_string(lstm_prediction))
            results_hbox.itemAt(7).widget().show()
        if self.algorithms[5]:
            self.arima_col_name.show()
            arima_prediction = self.controller.run_arima(ticker, self.hold_duration)
            results_hbox.itemAt(8).widget().setText(f"{self.result_to_string(arima_prediction)}"
                                                    f" +/- {self.controller.arima_confidences[self.hold_duration][ticker][1]:.2f}")
            results_hbox.itemAt(8).widget().show()

        volatility, volatility_category = self.controller.get_volatility(ticker, self.hold_duration)
        results_hbox.itemAt(9).widget().setText(f"{volatility:.2f} {volatility_category}")

        sharpe_ratio, sharpe_ratio_category = self.controller.get_sharpe_ratio(ticker, self.hold_duration)
        results_hbox.itemAt(10).widget().setText(f"{sharpe_ratio:.2f} {sharpe_ratio_category}")

        VaR = self.controller.get_VaR(ticker, self.hold_duration, volatility)
        results_hbox.itemAt(11).widget().setText(f"{VaR:.2f}")

        self.update_portfolio_results()
        self.tickers.append(ticker)

        results_hbox.addStretch(1)
        self.results_vbox.addLayout(results_hbox)
        self.results_map[ticker] = results_hbox

    def edit_stock(self, ticker, stock_name, one_share_price):
        is_long = self.controller.tickers_and_long_or_short[ticker]
        num_shares = self.controller.tickers_and_num_shares[ticker]
        popup = EditStockPopUp(ticker, stock_name, one_share_price, num_shares, is_long)
        popup.perform_change.connect(self.stock_change)
        popup.exec_()

    def delete_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.delete_layout(item.layout())
        if layout.parentWidget():
            layout.setParent(None)

    def stock_change(self, ticker, num_shares, investment, is_long):
        index = self.tickers.index(ticker)
        ticker_layout = self.results_vbox.itemAt(index + 1)
        if investment == -1:
            self.logger.info(f"Removing stock {ticker} from portfolio.")
            self.controller.remove_ticker(ticker)
            if ticker_layout.layout():
                self.delete_layout(ticker_layout.layout())
                self.results_vbox.removeItem(ticker_layout)
            self.tickers.remove(ticker)
            self.update_portfolio_results()
        else:
            self.logger.info(f"Stock {ticker} changed to: num_shares={num_shares}, is_long={is_long}.")
            self.controller.tickers_and_num_shares[ticker] = num_shares
            if is_long:
                long_short_str = "Long"
            else:
                long_short_str = "Short"
            self.results_map[ticker].itemAt(2).widget().setText("$ " + str(investment) + " " + long_short_str)
            algorithm_indices = [index for index, value in enumerate(self.algorithms) if value]
            self.controller.update_stock_info(ticker, num_shares, investment, is_long, algorithm_indices)
            for index in algorithm_indices:
                self.update_algorithm_values(index)


class AddStockPopUp(QDialog):
    valid_ticker_entered = pyqtSignal(str, float, int, bool)
    ticker = ""
    share_price = None
    should_validate = False

    def __init__(self, sp500_companies, top_50_esg_companies):
        super().__init__()
        self.setWindowTitle("Add Stock")

        self.sp500_companies = sp500_companies
        self.top_50_esg_companies = top_50_esg_companies

        layout = QVBoxLayout()

        search_by_hbox = QHBoxLayout()
        search_by_label = QLabel("Search by")
        self.by_name_radio = QRadioButton("name")
        self.by_name_radio.setChecked(True)
        self.by_name_radio.toggled.connect(self.search_by_radio_toggled)
        self.by_esg_radio = QRadioButton("top 50 best ESG score companies")
        self.by_esg_radio.toggled.connect(self.search_by_radio_toggled)
        search_by_hbox.addWidget(search_by_label)
        search_by_hbox.addWidget(self.by_name_radio)
        search_by_hbox.addWidget(self.by_esg_radio)

        ticker_label = QLabel("Please enter the stock ticker to add:")

        self.completer = QCompleter(sp500_companies, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)

        ticker_hbox = QHBoxLayout()
        self.ticker_name = CustomLineEdit(self.completer)
        self.ticker_name.returnPressed.connect(self.validate_ticker)
        self.ticker_name.setCompleter(self.completer)
        ticker_hbox.addWidget(self.ticker_name)
        self.ticker_enter_button = QPushButton("Enter")
        self.ticker_enter_button.clicked.connect(self.validate_ticker)
        ticker_hbox.addWidget(self.ticker_enter_button)

        self.invalid_ticker_label = QLabel("Invalid ticker entered")
        self.invalid_ticker_label.hide()
        self.invalid_ticker_label.setStyleSheet("color: red;")
        self.ticker_name.textChanged.connect(self.hide_invalid_labels)

        self.share_price_label = QLabel()
        self.share_price_label.hide()

        investment_hbox = QHBoxLayout()

        self.investment_label = QLabel("Enter the amount of shares:")
        self.investment_label.hide()

        self.investment_enter_button = QPushButton("Enter")
        self.investment_enter_button.clicked.connect(self.validate_investment)
        self.investment_enter_button.hide()

        self.investment = QLineEdit()
        self.investment.textChanged.connect(self.hide_invalid_investment_label)
        self.investment.mousePressEvent = self.investment_enter_button.setFocus()
        self.investment.returnPressed.connect(self.investment_entered)
        self.investment.hide()
        investment_hbox.addWidget(self.investment)
        investment_hbox.addWidget(self.investment_enter_button)

        self.invalid_investment_label = QLabel("The amount of shares has to be a positive integer.")
        self.invalid_investment_label.hide()
        self.invalid_investment_label.setStyleSheet("color: red;")
        self.ticker_name.textChanged.connect(self.hide_invalid_investment_label)

        long_short_layout = QHBoxLayout()
        self.investment_long = QRadioButton("Long")
        self.investment_long.setChecked(True)
        self.investment_long.hide()
        self.investment_short = QRadioButton("Short")
        self.investment_short.hide()
        long_short_layout.addWidget(self.investment_long)
        long_short_layout.addWidget(self.investment_short)

        buttons_hbox = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        buttons_hbox.addWidget(cancel_button)

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_ticker_to_portfolio)
        self.add_button.hide()
        buttons_hbox.addWidget(self.add_button)

        layout.addLayout(search_by_hbox)
        layout.addWidget(ticker_label)
        layout.addLayout(ticker_hbox)
        layout.addWidget(self.invalid_ticker_label)
        layout.addWidget(self.share_price_label)
        layout.addLayout(long_short_layout)
        layout.addWidget(self.investment_label)
        layout.addLayout(investment_hbox)
        layout.addWidget(self.invalid_investment_label)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def search_by_radio_toggled(self, checked):
        if checked:
            if self.by_name_radio.isChecked():
                newCompleterModel = QStringListModel(self.sp500_companies)
                self.completer.setModel(newCompleterModel)
            elif self.by_esg_radio.isChecked():
                newCompleterModel = QStringListModel(self.top_50_esg_companies)
                self.completer.setModel(newCompleterModel)

    def validate_ticker(self):
        ticker = self.ticker_name.text()
        is_valid, price = self.is_valid_ticker(ticker)

        if is_valid:
            self.ticker = ticker.split()[0]

            share_price = round(price.iloc[0], 2)
            self.share_price = share_price
            self.share_price_label.setText(f"Price of 1 share = ${share_price}")
            self.share_price_label.show()

            self.investment_label.show()
            self.investment.show()
            self.add_button.show()
            self.investment_long.show()
            self.investment_short.show()
            self.investment.setFocus()
            self.investment_enter_button.show()
        else:
            self.invalid_ticker_label.show()

    def is_valid_ticker(self, ticker):
        try:
            ticker = ticker.split()[0]
            if not re.match(r'^[A-Za-z0-9]+$', ticker):
                return False, -1
            data = yf.Ticker(ticker)
            one_day_data = data.history(period="1d")
            if len(one_day_data) > 0:
                return True, one_day_data["Close"]
            return False, -1
        except Exception:
            return False, -1

    def validate_investment(self):
        if not self.is_valid_investment(self.investment.text()):
            self.invalid_investment_label.show()
            self.investment.setFocus()

    def is_valid_investment(self, inv):
        return bool(re.match(r'^[1-9]\d*$', inv))

    def add_ticker_to_portfolio(self):
        num_shares = self.investment.text()
        if self.is_valid_investment(num_shares):
            if self.investment_long.isChecked():
                is_long = True
            else:
                is_long = False
            self.valid_ticker_entered.emit(self.ticker.upper(), self.share_price, int(num_shares), is_long)
            self.close()
        else:
            self.invalid_investment_label.show()
            self.investment.setFocus()

    def hide_invalid_labels(self):
        self.invalid_ticker_label.hide()
        self.invalid_investment_label.hide()

        self.share_price_label.hide()
        self.investment_long.hide()
        self.investment_short.hide()
        self.investment_label.hide()
        self.investment.hide()
        self.add_button.hide()

    def hide_invalid_investment_label(self):
        self.invalid_investment_label.hide()

    def investment_entered(self):
        self.add_button.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.accept()
            self.add_button.setFocus()
        else:
            super().keyPressEvent(event)


class CustomLineEdit(QLineEdit):
    enter_pressed = pyqtSignal()

    def __init__(self, completer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completer = completer

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.completer and self.completer.popup().isVisible():
                if self.completer.currentCompletion():
                    self.setText(self.completer.currentCompletion())
                else:
                    self.setText(self.completer.model().index(0, 0).data())
            self.enter_pressed.emit()
        else:
            super().keyPressEvent(event)


class EditStockPopUp(QDialog):
    perform_change = pyqtSignal(str, int, float, bool)

    def __init__(self, ticker, stock_name, one_share_price, num_shares, is_long):
        super().__init__()
        self.setWindowTitle(f"Edit {ticker} shares")

        self.ticker = ticker
        self.one_share_price = one_share_price
        self.is_long = is_long

        layout = QVBoxLayout()

        self.stock_name_label = QLabel(stock_name)
        self.share_price_label = QLabel(f"Price of 1 share = ${one_share_price}")

        investment_hbox = QHBoxLayout()

        self.investment_label = QLabel("Number of shares in portfolio:")
        investment_hbox.addWidget(self.investment_label)

        self.investment = QLineEdit(str(num_shares))
        self.investment.textChanged.connect(self.hide_invalid_investment_label)
        investment_hbox.addWidget(self.investment)

        self.invalid_investment_label = QLabel("The amount of shares has to be a positive integer.")
        self.invalid_investment_label.hide()
        self.invalid_investment_label.setStyleSheet("color: red;")

        long_short_layout = QHBoxLayout()
        self.investment_long = QRadioButton("Long")
        self.investment_short = QRadioButton("Short")
        if is_long:
            self.investment_long.setChecked(True)
        else:
            self.investment_short.setChecked(True)
        self.investment_long.toggled.connect(self.long_short_changed)
        self.investment_short.toggled.connect(self.long_short_changed)
        long_short_layout.addWidget(self.investment_long)
        long_short_layout.addWidget(self.investment_short)

        buttons_hbox = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        buttons_hbox.addWidget(cancel_button)

        self.save_changes_button = QPushButton("Save Changes")
        self.save_changes_button.clicked.connect(self.save_changes)
        self.save_changes_button.hide()
        buttons_hbox.addWidget(self.save_changes_button)

        self.delete_button = QPushButton(f"Delete {ticker} from portfolio")
        self.delete_button.clicked.connect(self.delete_stock)
        buttons_hbox.addWidget(self.delete_button)

        layout.addWidget(self.stock_name_label)
        layout.addWidget(self.share_price_label)
        layout.addLayout(investment_hbox)
        layout.addWidget(self.invalid_investment_label)
        layout.addLayout(long_short_layout)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def hide_invalid_investment_label(self):
        self.invalid_investment_label.hide()
        self.save_changes_button.show()

    def long_short_changed(self, checked):
        if checked:
            if self.investment_long.isChecked():
                self.is_long = True
            else:
                self.is_long = False
            self.save_changes_button.show()

    def save_changes(self):
        num_shares = self.investment.text()
        if bool(re.match(r'^[1-9]\d*$', num_shares)):
            self.perform_change.emit(self.ticker, int(num_shares), int(num_shares) * self.one_share_price, self.is_long)
            self.close()
        else:
            self.invalid_investment_label.show()

    def delete_stock(self):
        self.perform_change.emit(self.ticker, -1, -1, True)
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.accept()
        else:
            super().keyPressEvent(event)


class RankingTimeWarningPopUp(QDialog):
    decision_made = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Warning")

        layout = QVBoxLayout()

        # TODO: replace X with number (do this when I have good internet, maybe back in UK)
        warning_label = QLabel("Ranking process will take approximately X minutes.")
        warning_label_2 = QLabel("Do you still want to continue?")

        buttons_hbox = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.process_cancel_decision)
        buttons_hbox.addWidget(cancel_button)

        continue_button = QPushButton("Continue")
        continue_button.clicked.connect(self.process_continue_decision)
        buttons_hbox.addWidget(continue_button)

        layout.addWidget(warning_label)
        layout.addWidget(warning_label_2)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def process_cancel_decision(self):
        self.decision_made.emit(False)
        self.close()

    def process_continue_decision(self):
        self.decision_made.emit(True)
        self.close()

