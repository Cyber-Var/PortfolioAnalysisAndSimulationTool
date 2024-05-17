import os
import re
import traceback

from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox,
                             QScrollArea, QDialog, QLineEdit, QCompleter, QSizePolicy, QButtonGroup, QFrame,
                             QSpacerItem)
from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel, QUrl, QTimer
import yfinance as yf

from Page import Page


class PortfolioPage(QWidget, Page):
    """
    Portfolio analysis page
    """

    back_to_menu_page = pyqtSignal()
    open_single_stock_page = pyqtSignal(str, str, float, int, str, bool)

    portfolio_yellow_border_style = "font-weight: bold; border: 2px solid #F4FF96;"

    def __init__(self, main_window, controller, set_algorithms, hold_duration):
        """
        Constructor method
        :param main_window: main window of the app
        :param controller: the controller object
        :param set_algorithms: previously selected algorithms to display
        :param hold_duration: previously selected hold duration
        """
        super().__init__()

        self.main_window = main_window
        self.controller = controller
        self.set_algorithms = set_algorithms
        self.tickers = []

        # Initialize the radio button variables:

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
        self.algorithms = [False] * 5

        # Initialize the algorithms ranking box:

        self.outer_ranking_vbox = QVBoxLayout()
        self.outer_ranking_vbox.setContentsMargins(70, 30, 0, 0)

        self.ranking_vbox = QVBoxLayout()
        self.ranking_widget = QWidget()
        self.ranking_widget.setObjectName("rankingHBox")
        self.ranking_widget.setFixedSize(200, 250)
        self.ranking_widget.setLayout(self.ranking_vbox)

        self.outer_ranking_vbox.addWidget(self.ranking_widget)

        # Get and display previously saved ranking of algorithms:

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
        update_ranking_button.setStyleSheet("border-radius: 5px;")
        update_ranking_button.setFixedWidth(150)
        update_ranking_button.clicked.connect(self.play_cancel_sound)
        update_ranking_button.clicked.connect(self.show_ranking_time_warning_window)
        self.ranking_vbox.addWidget(update_ranking_button, alignment=Qt.AlignCenter)

        # Initialize the page's widgets:

        self.portfolio_col_names = []
        self.portfolio_results = []
        self.portfolio_label = None
        self.portfolio_amount = None
        self.portfolio_linear_regression = None
        self.portfolio_random_forest = None
        self.portfolio_bayesian = None
        self.portfolio_monte_carlo = None
        # self.portfolio_lstm = None
        self.portfolio_arima = None
        self.portfolio_volatility = None
        self.portfolio_volatility_category = None
        self.portfolio_sharpe_ratio = None
        self.portfolio_sharpe_category = None
        self.portfolio_VaR = None

        self.col_names = []
        self.result_col_names = []
        self.ticker_col_name = None
        self.stock_name_col_name = None
        self.amount_col_name = None
        self.lin_reg_col_name = None
        self.random_forest_col_name = None
        self.bayesian_col_name = None
        self.monte_carlo_col_name = None
        # self.lstm_col_name = None
        self.arima_col_name = None
        self.volatility_col_name = None
        self.sharpe_ratio_col_name = None
        self.VaR_col_name = None
        self.more_info_col_name = None
        self.edit_col_name = None

        self.results_vbox = None
        self.results_map = {}
        self.monte_profits_losses = {"1d": {}, "1w": {}, "1m": {}}

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Portfolio Analysis")

        self.build_page()

        self.setStyleSheet(self.load_stylesheet())

        self.setLayout(self.layout)

    def build_page(self):
        """
        Method for building the page
        """
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

        # Display the scrollable area with results:

        self.results_vbox = QVBoxLayout()

        column_names_hbox = QHBoxLayout()
        column_names_hbox.setSpacing(3)

        self.ticker_col_name = self.create_column_names_labels("Ticker")
        self.ticker_col_name.setFixedSize(60, 50)
        self.stock_name_col_name = self.create_column_names_labels("Name")
        self.stock_name_col_name.setFixedSize(160, 50)
        self.amount_col_name = self.create_column_names_labels("Investment\nAmount")
        self.amount_col_name.setFixedSize(130, 50)
        self.lin_reg_col_name = self.create_column_names_labels("Linear\nRegression")
        self.lin_reg_col_name.setFixedSize(90, 50)
        self.lin_reg_col_name.hide()
        self.random_forest_col_name = self.create_column_names_labels("Random\nForest")
        self.random_forest_col_name.setFixedSize(80, 50)
        self.random_forest_col_name.hide()
        self.bayesian_col_name = self.create_column_names_labels("Bayesian")
        self.bayesian_col_name.setFixedSize(130, 50)
        self.bayesian_col_name.hide()
        self.monte_carlo_col_name = self.create_column_names_labels("Monte Carlo\nSimulation")
        self.monte_carlo_col_name.setFixedSize(210, 50)
        self.monte_carlo_col_name.hide()
        # self.lstm_col_name = self.create_column_names_labels("LSTM")
        # self.lstm_col_name.setFixedSize(80, 50)
        # self.lstm_col_name.hide()
        self.arima_col_name = self.create_column_names_labels("ARIMA")
        self.arima_col_name.setFixedSize(130, 50)
        self.arima_col_name.hide()
        self.volatility_col_name = self.create_column_names_labels("Volatility")
        self.volatility_col_name.setFixedSize(110, 50)
        self.sharpe_ratio_col_name = self.create_column_names_labels("Sharpe\nRatio")
        self.sharpe_ratio_col_name.setFixedSize(110, 50)
        self.VaR_col_name = self.create_column_names_labels("Value\nat Risk")
        self.VaR_col_name.setFixedSize(70, 50)
        self.more_info_col_name = self.create_column_names_labels("More\nInfo")
        self.more_info_col_name.setFixedSize(50, 50)
        self.edit_col_name = self.create_column_names_labels("Edit\nStock")
        self.edit_col_name.setFixedSize(50, 50)

        self.col_names = [self.ticker_col_name, self.stock_name_col_name, self.amount_col_name,
                          self.volatility_col_name,
                          self.sharpe_ratio_col_name, self.VaR_col_name, self.more_info_col_name, self.edit_col_name]
        self.result_col_names = [self.lin_reg_col_name, self.random_forest_col_name, self.bayesian_col_name,
                                 self.monte_carlo_col_name, self.arima_col_name]

        for col_name_label in self.col_names:
            col_name_label.hide()

        column_names_hbox.addWidget(self.ticker_col_name)
        column_names_hbox.addWidget(self.stock_name_col_name)
        column_names_hbox.addWidget(self.amount_col_name)
        column_names_hbox.addWidget(self.lin_reg_col_name)
        column_names_hbox.addWidget(self.random_forest_col_name)
        column_names_hbox.addWidget(self.bayesian_col_name)
        column_names_hbox.addWidget(self.monte_carlo_col_name)
        # column_names_hbox.addWidget(self.lstm_col_name)
        column_names_hbox.addWidget(self.arima_col_name)
        column_names_hbox.addWidget(self.volatility_col_name)
        column_names_hbox.addWidget(self.sharpe_ratio_col_name)
        column_names_hbox.addWidget(self.VaR_col_name)
        column_names_hbox.addWidget(self.more_info_col_name)
        column_names_hbox.addWidget(self.edit_col_name)

        add_stock_button = QPushButton("+ Add Stock")
        add_stock_button.setObjectName("addStockButton")
        add_stock_button.setStyleSheet(
            "border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #AF40FF, stop:1 #00DDEB);")
        add_stock_button.setFixedSize(110, 50)
        add_stock_button.clicked.connect(self.play_cancel_sound)
        add_stock_button.clicked.connect(self.show_add_stock_window)

        column_names_hbox.addStretch(1)

        scrollable_area = QScrollArea()
        scrollable_area.setWidgetResizable(True)
        scrollable_widget = QWidget()
        scrollable_widget.setStyleSheet("border: 2px solid white;")
        scrollable_layout = QVBoxLayout()
        scrollable_layout.addLayout(column_names_hbox)
        scrollable_layout.addLayout(self.results_vbox)
        scrollable_layout.addWidget(add_stock_button, alignment=Qt.AlignCenter)
        scrollable_layout.addStretch(1)
        scrollable_widget.setLayout(scrollable_layout)
        scrollable_area.setWidget(scrollable_widget)

        back_button = QPushButton("Back")
        back_button.setObjectName("addStockButton")
        back_button.setFixedSize(90, 40)
        back_button.clicked.connect(self.play_cancel_sound)
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
        algorithms_widget.setFixedSize(350, 200)
        algorithms_widget.setLayout(algrithms_hbox)

        self.algorithms_label = QLabel("Algorithms:")
        self.algorithms_label.setObjectName('inputHeaderLabel')

        algorithms_vbox = QVBoxLayout()

        self.algorithm_1 = self.create_algorithm_checkbox("Linear Regression", 0)
        self.algorithm_2 = self.create_algorithm_checkbox("Random Forest", 1)
        self.algorithm_3 = self.create_algorithm_checkbox("Bayesian", 2)
        self.algorithm_4 = self.create_algorithm_checkbox("Monte Carlo Simulation", 3)
        # self.algorithm_5 = self.create_algorithm_checkbox("LSTM", 4)
        self.algorithm_5 = self.create_algorithm_checkbox("ARIMA", 4)

        algorithms_vbox.addWidget(self.algorithm_1)
        algorithms_vbox.addWidget(self.algorithm_2)
        algorithms_vbox.addWidget(self.algorithm_3)
        algorithms_vbox.addWidget(self.algorithm_4)
        algorithms_vbox.addWidget(self.algorithm_5)
        # algorithms_vbox.addWidget(self.algorithm_6)

        algrithms_hbox.addWidget(self.algorithms_label)
        algrithms_hbox.addLayout(algorithms_vbox)

        input_hbox = QHBoxLayout()
        input_hbox.setSpacing(170)
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
        """
        Method for building the results of the overall portfolio
        """
        portfolio_hbox = QHBoxLayout()
        portfolio_hbox.setSpacing(3)

        self.portfolio_label = QLabel("Overall Portfolio results")
        self.portfolio_label.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_label.setFixedSize(223, 70)

        self.portfolio_amount = QLabel("-")
        self.portfolio_amount.setObjectName("portfolioResultLabel")
        self.portfolio_amount.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_amount.setFixedSize(130, 70)

        self.portfolio_linear_regression = QLabel("")
        self.portfolio_linear_regression.setFixedSize(90, 70)
        self.portfolio_random_forest = QLabel("")
        self.portfolio_random_forest.setFixedSize(80, 70)
        self.portfolio_bayesian = QLabel("")
        self.portfolio_bayesian.setFixedSize(130, 70)
        self.portfolio_monte_carlo = QLabel("")
        self.portfolio_monte_carlo.setFixedSize(210, 70)
        # self.portfolio_lstm = QLabel("")
        # self.portfolio_lstm.setFixedSize(80, 70)
        self.portfolio_arima = QLabel("")
        self.portfolio_arima.setFixedSize(130, 70)

        self.portfolio_linear_regression.setObjectName("resultLabel")
        self.portfolio_linear_regression.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_random_forest.setObjectName("resultLabel")
        self.portfolio_random_forest.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_bayesian.setObjectName("resultLabel")
        self.portfolio_bayesian.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_monte_carlo.setObjectName("resultLabel")
        self.portfolio_monte_carlo.setStyleSheet(self.portfolio_yellow_border_style)
        # self.portfolio_lstm.setObjectName("resultLabel")
        # self.portfolio_lstm.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_arima.setObjectName("resultLabel")
        self.portfolio_arima.setStyleSheet(self.portfolio_yellow_border_style)

        self.portfolio_linear_regression.setFixedHeight(70)
        self.portfolio_random_forest.setFixedHeight(70)
        self.portfolio_bayesian.setFixedHeight(70)
        self.portfolio_monte_carlo.setFixedHeight(70)
        # self.portfolio_lstm.setFixedHeight(70)
        self.portfolio_arima.setFixedHeight(70)

        self.portfolio_linear_regression.hide()
        self.portfolio_random_forest.hide()
        self.portfolio_bayesian.hide()
        self.portfolio_monte_carlo.hide()
        # self.portfolio_lstm.hide()
        self.portfolio_arima.hide()

        self.portfolio_vol_frame, portfolio_vol_frame_layout = self.create_frame()
        self.portfolio_vol_frame.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_volatility = QLabel()
        self.portfolio_volatility.setFixedSize(50, 30)
        # self.portfolio_volatility.setStyleSheet("color: white; font-weight: normal; border-width: 0;")
        self.portfolio_volatility.setStyleSheet("color: white; border-width: 0;")
        self.portfolio_volatility_category = QLabel()
        self.portfolio_volatility_category.setFixedSize(50, 30)
        self.portfolio_volatility_category.setAlignment(Qt.AlignCenter)
        portfolio_vol_frame_layout.addWidget(self.portfolio_volatility)
        portfolio_vol_frame_layout.addWidget(self.portfolio_volatility_category)
        self.portfolio_vol_frame.setFixedSize(110, 70)

        self.portfolio_sharpe_frame, portfolio_sharpe_frame_layout = self.create_frame()
        self.portfolio_sharpe_frame.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_sharpe_ratio = QLabel()
        self.portfolio_sharpe_ratio.setFixedSize(50, 30)
        # self.portfolio_sharpe_ratio.setStyleSheet("color: white; font-weight: normal; border-width: 0;")
        self.portfolio_sharpe_ratio.setStyleSheet("color: white; border-width: 0;")
        self.portfolio_sharpe_category = QLabel()
        self.portfolio_sharpe_category.setFixedSize(50, 30)
        self.portfolio_sharpe_category.setAlignment(Qt.AlignCenter)
        portfolio_sharpe_frame_layout.addWidget(self.portfolio_sharpe_ratio)
        portfolio_sharpe_frame_layout.addWidget(self.portfolio_sharpe_category)
        self.portfolio_sharpe_frame.setFixedSize(110, 70)

        self.portfolio_VaR = QLabel("")
        self.portfolio_VaR.setObjectName("resultLabel")
        self.portfolio_VaR.setStyleSheet(self.portfolio_yellow_border_style)
        self.portfolio_VaR.setFixedSize(70, 70)

        self.portfolio_more_info_button = QPushButton("--->")
        self.portfolio_more_info_button.setFixedSize(50, 70)
        self.portfolio_more_info_button.setObjectName("portfolioButton")
        self.portfolio_more_info_button.setDisabled(True)
        self.portfolio_more_info_button.clicked.connect(self.play_cancel_sound)
        self.portfolio_more_info_button.setStyleSheet(
            "background-color: #333; color: #AAA; border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #67497E, stop:1 #09828A);")

        self.portfolio_edit_button = QPushButton("Edit")
        self.portfolio_edit_button.setFixedSize(50, 70)
        self.portfolio_edit_button.setObjectName("portfolioButton")
        self.portfolio_edit_button.setDisabled(True)
        self.portfolio_edit_button.clicked.connect(self.play_cancel_sound)
        self.portfolio_edit_button.setStyleSheet(
            "background-color: #333; color: #AAA; border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #67497E, stop:1 #09828A);")

        self.portfolio_col_names = [self.portfolio_label, self.portfolio_amount, self.portfolio_vol_frame,
                                    self.portfolio_sharpe_frame, self.portfolio_VaR]
        self.portfolio_results = [self.portfolio_linear_regression, self.portfolio_random_forest,
                                  self.portfolio_bayesian, self.portfolio_monte_carlo,
                                  self.portfolio_arima]

        for col_name_label in self.portfolio_col_names:
            col_name_label.hide()
        self.portfolio_more_info_button.hide()
        self.portfolio_edit_button.hide()

        portfolio_hbox.addWidget(self.portfolio_label)
        portfolio_hbox.addWidget(self.portfolio_amount)
        portfolio_hbox.addWidget(self.portfolio_linear_regression)
        portfolio_hbox.addWidget(self.portfolio_random_forest)
        portfolio_hbox.addWidget(self.portfolio_bayesian)
        portfolio_hbox.addWidget(self.portfolio_monte_carlo)
        # portfolio_hbox.addWidget(self.portfolio_lstm)
        portfolio_hbox.addWidget(self.portfolio_arima)
        portfolio_hbox.addWidget(self.portfolio_vol_frame)
        portfolio_hbox.addWidget(self.portfolio_sharpe_frame)
        portfolio_hbox.addWidget(self.portfolio_VaR)
        portfolio_hbox.addWidget(self.portfolio_more_info_button)
        portfolio_hbox.addWidget(self.portfolio_edit_button)

        portfolio_hbox.addStretch(1)
        self.results_vbox.addLayout(portfolio_hbox)

    def create_hold_duration_button(self, name):
        """
        Method for creating one hold duration radio button
        :param name: name of the button
        :return: the radio button
        """
        button = QRadioButton(name)
        button.setObjectName('inputLabel')
        button.toggled.connect(self.play_radio_sound)
        button.toggled.connect(self.hold_duration_button_toggled)
        return button

    def create_algorithm_checkbox(self, name, ii):
        """
        Method for creating one algorithm checkbox button
        :param name: name of the button
        :return: the checkbox button
        """
        button = QCheckBox(name)
        button.setObjectName('inputLabel')
        button.stateChanged.connect(self.play_radio_sound)
        button.stateChanged.connect(lambda state, index=ii: self.algorithms_state_changed(state, index))
        if self.set_algorithms[ii]:
            button.setChecked(True)
        return button

    def create_column_names_labels(self, name):
        """
        Method for creating one column name label
        :param name: name of the label
        :return: the label
        """
        label = QLabel(name)
        label.setObjectName('columnNameLabel')
        return label

    def create_frame(self):
        """
        Method for creating one frame for label with multiple styles inside
        :return: the frame
        """
        frame = QFrame()
        frame_layout = QHBoxLayout(frame)
        frame_layout.setSpacing(0)
        frame.setObjectName("resultLabel")
        return frame, frame_layout

    def hold_duration_button_toggled(self, checked):
        """
        Method that is called when a new hold duration is selected
        :param checked: whether the radio button is checked
        """
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
        """
        Method for displaying new algorithms ranking
        """
        for alg_index, algorithm in enumerate(self.rankings[self.hold_duration]):
            ranking_label = QLabel(algorithm)
            ranking_label.setObjectName('rankingLabel')
            self.ranking_vbox.itemAt(alg_index + 1).widget().setText(f"{alg_index + 1}. {algorithm}")

    def result_to_string(self, result):
        """
        Method for converting an algorithmic prediction in a human-readable format
        :param result: the prediction
        :return: prediction in human-readable format
        """
        if result >= 0:
            return f"+${result:.2f}", True
        return f"-${abs(result):.2f}", False

    def algorithms_state_changed(self, state, index):
        """
        Method that is called when the user selects (or unselects) an algorithm
        :param state: selected or unselected
        :param index: index of the algorithm
        """
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
        """
        Method for displaying new algorithmic results
        :param index: index of the algorithm
        """
        self.logger.info('Updating the Algorithmic results')

        try:
            # Loop through each stock in portfolio:
            algorithm_name = self.controller.algorithms_with_indices[index]
            algorithmic_results = self.controller.results[algorithm_name][self.hold_duration]
            for ticker in self.controller.tickers_and_investments.keys():
                label = self.results_map[ticker].itemAt(3 + index).widget()

                # Run the algorithm if necessary:
                if ticker not in algorithmic_results.keys():
                    self.controller.run_algorithm(ticker, index, self.hold_duration)

                # Display LR, RF, BRR or ARIMA prediction:
                if index != 3:
                    str_result, is_green = self.result_to_string(algorithmic_results[ticker])

                    if index == 2:
                        label_text = f"{str_result} +/- {self.format_number(abs(self.controller.bayesian_confidences[self.hold_duration][ticker][1]))}"
                    elif index == 4:
                        label_text = f"{str_result} +/- {self.format_number(abs(self.controller.arima_confidences[self.hold_duration][ticker][1]))}"
                    else:
                        label_text = str_result
                # Display MC prediction:
                else:
                    label_text = algorithmic_results[ticker]
                    growth_fall_split = label_text.split()
                    growth_fall = growth_fall_split[-1]
                    is_long = self.controller.tickers_and_long_or_short[ticker]
                    if (growth_fall == "growth" and is_long) or (growth_fall == "fall" and not is_long):
                        label_text = " ".join(growth_fall_split[:-1]) + " profit"
                        is_green = True
                        self.monte_profits_losses[self.hold_duration][ticker] = True
                    else:
                        label_text = " ".join(growth_fall_split[:-1]) + " loss"
                        is_green = False
                        self.monte_profits_losses[self.hold_duration][ticker] = False
                label.setText(label_text)
                # Set the color to green for profit or red for loss:
                if is_green:
                    self.results_map[ticker].itemAt(3 + index).widget().setStyleSheet("color: green;")
                else:
                    self.results_map[ticker].itemAt(3 + index).widget().setStyleSheet("color: red;")
                label.show()
                self.result_col_names[index].show()

                volatility, volatility_category = self.controller.get_volatility(ticker, self.hold_duration)
                volatility_labels = self.results_map[ticker].itemAt(8).widget().findChildren(QLabel)
                volatility_labels[0].setText(f"{volatility:.2f} ")
                volatility_labels[1].setText(volatility_category)
                volatility_labels[1].setStyleSheet(self.process_category_style(volatility_category))

                sharpe_ratio, sharpe_ratio_category = self.controller.get_sharpe_ratio(ticker, self.hold_duration)
                sharpe_labels = self.results_map[ticker].itemAt(9).widget().findChildren(QLabel)
                sharpe_labels[0].setText(f"{sharpe_ratio:.2f} ")
                sharpe_labels[1].setText(sharpe_ratio_category)
                sharpe_labels[1].setStyleSheet(self.process_category_style(sharpe_ratio_category))

                VaR = self.controller.get_VaR(ticker, self.hold_duration, volatility)
                self.results_map[ticker].itemAt(10).widget().setText(f"${VaR:.2f}")

                # Call the re-calculation of results for overall portfolio:
                self.update_portfolio_results()
        except:
            self.logger.error("Error occurred when updating algorithmic results.")
            self.main_window.show_error_window("Error occurred when updating algorithmic results.",
                                               "Please check your Internet connection.")

    def update_portfolio_results(self):
        """
        Method for updating the results of overall portfolio
        """
        self.logger.info('Updating the Portfolio results')

        try:
            # Show and hide labels as needed:

            if len(self.controller.tickers_and_investments) == 0:
                for col_name_label in self.portfolio_col_names:
                    col_name_label.hide()
                self.portfolio_more_info_button.hide()
                self.portfolio_edit_button.hide()

                for col_name_label in self.result_col_names:
                    col_name_label.hide()

                for index, is_chosen in enumerate(self.algorithms):
                    if is_chosen:
                        self.portfolio_results[index].hide()

                if self.ticker_col_name.isVisible():
                    for col_name_label in self.col_names:
                        col_name_label.hide()
                    for col_name_label in self.result_col_names:
                        col_name_label.hide()
                return

            if not self.ticker_col_name.isVisible():
                for i in range(len(self.col_names)):
                    self.col_names[i].show()

            for col_name_label in self.portfolio_col_names:
                col_name_label.show()
            self.portfolio_more_info_button.show()
            self.portfolio_edit_button.show()

            # Calculate portfolio outcome of each algorithm:
            long_amount = 0
            short_amount = 0
            for ticker in self.controller.tickers_and_investments.keys():
                if self.controller.tickers_and_long_or_short[ticker]:
                    long_amount += self.controller.tickers_and_investments[ticker]
                else:
                    short_amount += self.controller.tickers_and_investments[ticker]
            self.portfolio_amount.setText(f"${long_amount:.2f} Long\n${short_amount:.2f} Short")
            for index, is_chosen in enumerate(self.algorithms):
                if is_chosen:
                    # Show the MC prediction:
                    if index == 3:
                        result = self.controller.calculate_portfolio_monte_carlo(self.hold_duration)
                        monte_profits = list(self.monte_profits_losses[self.hold_duration].values()).count(True)
                        monte_losses = len(self.monte_profits_losses[self.hold_duration]) - monte_profits
                        if monte_profits >= monte_losses:
                            result = " ".join(result.split()[:-1]) + " profit"
                            is_green = True
                        else:
                            result = " ".join(result.split()[:-1]) + " loss"
                            is_green = False
                    # Show LR, RF, BRR or ARIMA prediction:
                    else:
                        num_result = self.controller.calculate_portfolio_result(index, self.hold_duration)
                        result, is_green = self.result_to_string(num_result)
                    if index == 2 or index == 4:
                        confidence = self.controller.calculate_portfolio_confidence(index, self.hold_duration)
                        result += f"+/- {self.format_number(abs(confidence))}"

                    # Set color to green for profit or red for loss:
                    self.portfolio_results[index].setText(result)
                    if is_green:
                        self.portfolio_results[index].setStyleSheet(
                            f"font-weight: bold; color: green; {self.portfolio_yellow_border_style}")
                    else:
                        self.portfolio_results[index].setStyleSheet(
                            f"font-weight: bold; color: red; {self.portfolio_yellow_border_style}")
                    self.portfolio_results[index].show()
                else:
                    self.portfolio_results[index].hide()

            # Update portfolio risk metrics:
            ticker_keys = self.controller.tickers_and_investments.keys()
            if len(ticker_keys) == 1:
                ticker = list(ticker_keys)[0]
                portfolio_vol, portfolio_vol_cat = self.controller.volatilities[ticker]
                portfolio_sharpe, portfolio_share_cat = self.controller.sharpe_ratios[ticker]
                portfolio_VaR = self.controller.VaRs[ticker]
            else:
                portfolio_vol, portfolio_vol_cat = self.controller.get_portfolio_volatility(self.hold_duration)
                portfolio_sharpe, portfolio_share_cat = self.controller.get_portfolio_sharpe_ratio(self.hold_duration,
                                                                                                   portfolio_vol)
                portfolio_VaR = self.controller.get_portfolio_VaR(self.hold_duration, portfolio_vol)

            self.portfolio_volatility.setText(f"{portfolio_vol:.2f}")
            self.portfolio_volatility_category.setText(portfolio_vol_cat)
            self.portfolio_volatility_category.setStyleSheet(self.process_category_style(portfolio_vol_cat))

            self.portfolio_sharpe_ratio.setText(f"{portfolio_sharpe:.2f}")
            self.portfolio_sharpe_category.setText(portfolio_share_cat)
            self.portfolio_sharpe_category.setStyleSheet(self.process_category_style(portfolio_share_cat))

            self.portfolio_VaR.setText(f"${portfolio_VaR:.2f}")
        except:
            traceback.print_exc()
            self.main_window.show_error_window("Error occurred when updating algorithmic results.",
                                               "Please check your Internet connection.")

    def format_number(self, num):
        """
        Method for rounding numeric results
        :param num: number to format
        :return: formatted number
        """
        if abs(num) < 0.01:
            return f"{num:.4f}"
        else:
            return f"{num:.2f}"

    def process_category_style(self, category):
        """
        Method that returns CSS style of icon
        :param category: Low/Normal/High
        :return: the CSS style of icon
        """
        if category == "Low":
            # 76D7C4
            return "font-size: 12px; color: #8EF9F3; border: 2px solid #8EF9F3; border-radius: 5px;"
        elif category == "Normal":
            return "font-size: 12px; color: #A882DD; border: 2px solid #A882DD; border-radius: 5px;"
        return "font-size: 12px; color: #FF5733; border: 2px solid #FF5733; border-radius: 5px;"

    def update_shares_results(self):
        """
        Method that calls update of each algorithm
        """
        try:
            for index, is_chosen in enumerate(self.algorithms):
                if is_chosen:
                    self.update_algorithm_values(index)
        except:
            traceback.print_exc()
            self.main_window.show_error_window("Error occurred when updating algorithmic results.",
                                               "Please check your Internet connection.")

    def show_add_stock_window(self):
        """
        Method that displays pop-up for adding a new stock
        """
        popup = AddStockPopUp(self.main_window, self.controller.get_above_1_bil_tickers(),
                              self.controller.get_top_100_esg_companies(), self.controller)
        popup.valid_ticker_entered.connect(self.add_ticker)
        popup.exec_()

    def show_ranking_time_warning_window(self):
        """
        Method that displays pop-up with warning that algorithms ranking is timely
        """
        popup = RankingTimeWarningPopUp()
        popup.decision_made.connect(self.process_ranking_request)
        popup.exec_()

    def process_ranking_request(self, should_process):
        """
        Method that calls an update of algorithms ranking
        :param should_process: boolean
        """
        if should_process:
            try:
                self.rankings = self.controller.handle_ranking(True)
                self.update_ranking_display()
            except:
                self.main_window.show_error_window("Error occurred when ranking algorithms.",
                                                   "Please check your Internet connection.")

    widths = [60, 160, 130, 90, 80, 130, 210, 130, 110, 110, 70]

    def add_ticker(self, ticker, one_share_price, num_shares, is_long, not_initial=True):
        """
        Method that adds a stock to portfolio
        :param ticker: stock's ticker
        :param one_share_price: price of 1 share
        :param num_shares: number of shares selected
        :param is_long: short/long investment
        :param not_initial: boolean
        """
        self.logger.info('Adding new stock to portfolio.')

        try:
            investment = round(one_share_price * num_shares, 2)

            if not_initial:
                self.controller.add_ticker(ticker, num_shares, investment, is_long)

            # Update the display:

            results_hbox = QHBoxLayout()
            results_hbox.setSpacing(3)

            for i in range(11):
                if 8 <= i <= 9:
                    frame, frame_layout = self.create_frame()
                    label = QLabel()
                    label.setFixedSize(50, 25)
                    label.setStyleSheet("color: white; font-weight: normal; border-width: 0;")
                    label_category = QLabel()
                    label_category.setFixedSize(50, 25)
                    label_category.setAlignment(Qt.AlignCenter)
                    frame_layout.addWidget(label)
                    frame_layout.addWidget(label_category)
                    frame.setFixedSize(self.widths[i], 50)
                    results_hbox.addWidget(frame)
                else:
                    label = QLabel()
                    if i != 0:
                        label.setObjectName("resultLabel")
                    label.setFixedSize(self.widths[i], 50)
                    if 3 <= i <= 7:
                        label.hide()
                    results_hbox.addWidget(label)

            results_hbox.itemAt(0).widget().setText(ticker)

            stock_info = yf.Ticker(ticker).info
            stock_name = stock_info.get('longName', 'N/A')
            results_hbox.itemAt(1).widget().setText(stock_name)

            if is_long:
                long_short_str = "Long"
            else:
                long_short_str = "Short"
            results_hbox.itemAt(2).widget().setText("$" + str(investment) + " " + long_short_str)

            # Run and display algorithmic predictions:

            if self.algorithms[0]:
                self.lin_reg_col_name.show()
                lin_reg_prediction = self.controller.run_linear_regression(ticker, self.hold_duration)
                result, is_green = self.result_to_string(lin_reg_prediction)
                if is_green:
                    results_hbox.itemAt(3).widget().setObjectName("greenResultLabel")
                else:
                    results_hbox.itemAt(3).widget().setObjectName("redResultLabel")
                results_hbox.itemAt(3).widget().setText(result)
                results_hbox.itemAt(3).widget().show()
            if self.algorithms[1]:
                self.random_forest_col_name.show()
                random_forest_prediction = self.controller.run_random_forest(ticker, self.hold_duration)
                result, is_green = self.result_to_string(random_forest_prediction)
                if is_green:
                    results_hbox.itemAt(4).widget().setObjectName("greenResultLabel")
                else:
                    results_hbox.itemAt(4).widget().setObjectName("redResultLabel")
                results_hbox.itemAt(4).widget().setText(result)
                results_hbox.itemAt(4).widget().show()
            if self.algorithms[2]:
                self.bayesian_col_name.show()
                bayesian_prediction = self.controller.run_bayesian(ticker, self.hold_duration)
                result, is_green = self.result_to_string(bayesian_prediction[0][0])
                if is_green:
                    results_hbox.itemAt(5).widget().setObjectName("greenResultLabel")
                else:
                    results_hbox.itemAt(5).widget().setObjectName("redResultLabel")
                results_hbox.itemAt(5).widget().setText(f"{result}"
                                                        f" +/- {self.format_number(abs(self.controller.bayesian_confidences[self.hold_duration][ticker][1]))}")
                results_hbox.itemAt(5).widget().show()
            if self.algorithms[3]:
                self.monte_carlo_col_name.show()
                monte_carlo_prediction = self.controller.run_monte_carlo(ticker, self.hold_duration)
                growth_fall = monte_carlo_prediction.split()
                if (growth_fall[-1] == "growth" and is_long) or (growth_fall[-1] == "fall" and not is_long):
                    monte_carlo_prediction = " ".join(growth_fall[:-1]) + " profit"
                    results_hbox.itemAt(6).widget().setObjectName("greenResultLabel")
                    self.monte_profits_losses[self.hold_duration][ticker] = True
                else:
                    monte_carlo_prediction = " ".join(growth_fall[:-1]) + " loss"
                    results_hbox.itemAt(6).widget().setObjectName("redResultLabel")
                    self.monte_profits_losses[self.hold_duration][ticker] = False
                results_hbox.itemAt(6).widget().setText(monte_carlo_prediction)
                results_hbox.itemAt(6).widget().show()
            # if self.algorithms[4]:
            #     self.lstm_col_name.show()
            #     lstm_prediction = self.controller.run_lstm(ticker, self.hold_duration)
            #     result, is_green = self.result_to_string(lstm_prediction)
            #     if is_green:
            #         results_hbox.itemAt(7).widget().setObjectName("greenResultLabel")
            #     else:
            #         results_hbox.itemAt(7).widget().setObjectName("redResultLabel")
            #     results_hbox.itemAt(7).widget().setText(result)
            #     results_hbox.itemAt(7).widget().show()
            if self.algorithms[4]:
                self.arima_col_name.show()
                arima_prediction = self.controller.run_arima(ticker, self.hold_duration)
                result, is_green = self.result_to_string(arima_prediction)
                if is_green:
                    results_hbox.itemAt(7).widget().setObjectName("greenResultLabel")
                else:
                    results_hbox.itemAt(7).widget().setObjectName("redResultLabel")
                results_hbox.itemAt(7).widget().setText(f"{result}"
                                                        f" +/- {self.format_number(abs(self.controller.arima_confidences[self.hold_duration][ticker][1]))}")
                results_hbox.itemAt(7).widget().show()

            # Calculate and display risk metrics:

            volatility, volatility_category = self.controller.get_volatility(ticker, self.hold_duration)
            volatility_labels = results_hbox.itemAt(8).widget().findChildren(QLabel)
            volatility_labels[0].setText(f"{volatility:.2f} ")
            volatility_labels[1].setText(volatility_category)
            volatility_labels[1].setStyleSheet(self.process_category_style(volatility_category))

            sharpe_ratio, sharpe_ratio_category = self.controller.get_sharpe_ratio(ticker, self.hold_duration)
            sharpe_labels = results_hbox.itemAt(9).widget().findChildren(QLabel)
            sharpe_labels[0].setText(f"{sharpe_ratio:.2f} ")
            sharpe_labels[1].setText(sharpe_ratio_category)
            sharpe_labels[1].setStyleSheet(self.process_category_style(sharpe_ratio_category))

            VaR = self.controller.get_VaR(ticker, self.hold_duration, volatility)
            results_hbox.itemAt(10).widget().setText(f"${VaR:.2f}")

            more_info_button = QPushButton("--->")
            more_info_button.setFixedSize(50, 50)
            more_info_button.clicked.connect(self.play_cancel_sound)
            more_info_button.clicked.connect(
                lambda: self.open_single_stock_page.emit(ticker, stock_name, one_share_price,
                                                         num_shares, self.hold_duration,
                                                         is_long))
            more_info_button.setObjectName("portfolioButton")
            more_info_button.setStyleSheet(
                "#portfolioButton { border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #AF40FF, stop:1 #00DDEB);}")
            results_hbox.addWidget(more_info_button)

            edit_button = QPushButton("Edit")
            edit_button.setFixedSize(50, 50)
            edit_button.clicked.connect(self.play_cancel_sound)
            edit_button.clicked.connect(lambda: self.edit_stock(ticker, stock_name, one_share_price))
            edit_button.setObjectName("portfolioButton")
            edit_button.setStyleSheet(
                "#portfolioButton { border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #AF40FF, stop:1 #00DDEB);}")
            results_hbox.addWidget(edit_button)

            self.update_portfolio_results()
            self.tickers.append(ticker)

            results_hbox.addStretch(1)
            self.results_vbox.addLayout(results_hbox)
            self.results_map[ticker] = results_hbox

        except Exception:
            self.logger.error("Unable to add new ticker.")
            traceback.print_exc()
            self.main_window.show_error_window("Error occurred when adding the stock to portfolio.",
                                               "Please check your Internet connection.")

    def edit_stock(self, ticker, stock_name, one_share_price):
        """
        Method that displays pop-up for updating stock info
        :param ticker: ticker of stock to update
        :param stock_name: name of the stock
        :param one_share_price: price of 1 share
        :return:
        """
        is_long = self.controller.tickers_and_long_or_short[ticker]
        num_shares = self.controller.tickers_and_num_shares[ticker]
        popup = EditStockPopUp(ticker, stock_name, one_share_price, num_shares, is_long)
        popup.perform_change.connect(self.stock_change)
        popup.exec_()

    def delete_layout(self, layout):
        """
        Method for deleting a layout and all its widgets
        :param layout: the layout
        """
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.delete_layout(item.layout())
        if layout.parentWidget():
            layout.setParent(None)

    def stock_change(self, ticker, num_shares, investment, is_long):
        """
        Method for changing stock info or deleting stock as requested by user
        :param ticker: ticker of the stock to edit
        :param num_shares: number of shares requested
        :param investment: investment amount defined
        :param is_long: long/short investment
        """
        try:
            index = self.tickers.index(ticker)
            ticker_layout = self.results_vbox.itemAt(index + 1)

            # If delete is requested, remove stock from portfolio:
            if investment == -1:
                self.logger.info(f"Removing stock {ticker} from portfolio.")
                self.controller.remove_ticker(ticker)
                if ticker_layout.layout():
                    self.delete_layout(ticker_layout.layout())
                    self.results_vbox.removeItem(ticker_layout)
                self.tickers.remove(ticker)
                self.update_portfolio_results()
            # Otherwise, update stock info:
            else:
                if num_shares == self.controller.tickers_and_num_shares[ticker] and \
                        self.controller.tickers_and_long_or_short[ticker] == is_long:
                    return

                self.logger.info(f"Stock {ticker} changed to: num_shares={num_shares}, is_long={is_long}.")

                if is_long:
                    long_short_str = "Long"
                else:
                    long_short_str = "Short"
                self.results_map[ticker].itemAt(2).widget().setText(f"${investment:.2f} {long_short_str}")
                algorithm_indices = [index for index, value in enumerate(self.algorithms) if value]

                only_change_sign = (num_shares == self.controller.tickers_and_num_shares[ticker] and
                                    self.controller.tickers_and_long_or_short[ticker] != is_long)
                self.controller.update_stock_info(ticker, num_shares, investment, is_long, algorithm_indices,
                                                  only_change_sign)
                for index in algorithm_indices:
                    self.update_algorithm_values(index)
        except Exception:
            traceback.print_exc()
            if investment == -1:
                self.logger.error("Unable to remove stock.")
                self.main_window.show_error_window("Error occurred when deleting the stock from portfolio.",
                                                   "Please check your Internet connection.")
            else:
                self.logger.error("Unable to edit stock.")
                self.main_window.show_error_window("Error occurred when editing the stock in portfolio.",
                                                   "Please check your Internet connection.")


class AddStockPopUp(QDialog):
    """
    Pop-up for adding a new stock to portfolio
    """
    valid_ticker_entered = pyqtSignal(str, float, int, bool)
    ticker = ""
    share_price = None
    should_validate = False

    def __init__(self, main_window, market_cap_above_billion_companies, top_100_esg_companies, controller):
        """
        Constructor method
        :param main_window: main window of the application
        :param market_cap_above_billion_companies: list of companies with market capitalisation above 1 billion dollars
        :param top_100_esg_companies: top 100 companies by ESG scores
        :param controller: the controller object
        """
        super().__init__()
        self.setWindowTitle("Add Stock to Portfolio")

        self.main_window = main_window
        self.controller = controller
        self.sp500_companies = market_cap_above_billion_companies
        self.top_100_esg_companies = top_100_esg_companies
        self.current_list = market_cap_above_billion_companies

        # Read the stylesheet file:
        path = self.get_file_path("style.css")
        with open(path, "r") as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        button_save_style = (
            "QPushButton:hover {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #12CB0B, stop:1 #0EC1D0);"
            "}"
            "QPushButton:pressed {"
            " background-color: black;"
            "}")

        # Set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')
        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')
        radio_sound_path = os.path.join(sound_directory, 'radio_button.wav')

        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()
        self.sound_radio = QSoundEffect()

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))
        self.sound_radio.setSource(QUrl.fromLocalFile(radio_sound_path))

        # Build the page:

        layout = QVBoxLayout()

        ticker_vbox = QVBoxLayout()
        ticker_widget = QWidget()
        ticker_widget.setObjectName("addStockVBox")
        ticker_widget.setFixedSize(500, 200)
        ticker_widget.setLayout(ticker_vbox)

        search_by_hbox = QHBoxLayout()
        search_by_label = QLabel("Search within:")
        search_by_label.setObjectName("addStockLabel")
        self.by_name_radio = QRadioButton("companies with capitalisation\nabove 1 billion dollars")
        self.by_name_radio.setChecked(True)
        self.by_name_radio.toggled.connect(self.sound_radio.play)
        self.by_name_radio.toggled.connect(self.search_by_radio_toggled)
        self.by_esg_radio = QRadioButton("top 100 companies\nby ESG score")
        self.by_esg_radio.toggled.connect(self.sound_radio.play)
        self.by_esg_radio.toggled.connect(self.search_by_radio_toggled)
        self.search_by_group = QButtonGroup(self)
        self.search_by_group.addButton(self.by_name_radio)
        self.search_by_group.addButton(self.by_esg_radio)
        search_by_hbox.addWidget(search_by_label)
        search_by_hbox.addWidget(self.by_name_radio)
        search_by_hbox.addWidget(self.by_esg_radio)

        ticker_label = QLabel("Please enter the stock ticker to add:")
        ticker_label.setObjectName("addStockLabel")

        self.completer = QCompleter(market_cap_above_billion_companies, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)

        ticker_hbox = QHBoxLayout()
        self.ticker_name = CustomLineEdit(self.completer)
        self.ticker_name.setFixedSize(300, 30)
        self.ticker_name.returnPressed.connect(self.validate_ticker)
        self.ticker_name.setCompleter(self.completer)
        self.ticker_enter_button = QPushButton("Enter")
        self.ticker_enter_button.setFixedSize(50, 35)
        self.ticker_enter_button.clicked.connect(self.validate_ticker)

        ticker_hbox.addStretch(1)
        ticker_hbox.setSpacing(10)
        ticker_hbox.addWidget(self.ticker_name)
        ticker_hbox.addWidget(self.ticker_enter_button)
        ticker_hbox.addStretch(1)

        self.invalid_ticker_label = QLabel("Please choose a company from list")
        self.invalid_ticker_label.hide()
        self.invalid_ticker_label.setStyleSheet("font-size: 15px; color: #C70C0C;")
        self.ticker_name.textChanged.connect(self.hide_invalid_labels)

        investment_vbox = QVBoxLayout()
        self.investment_widget = QWidget()
        self.investment_widget.setObjectName("addStockVBox")
        self.investment_widget.setFixedSize(500, 200)
        self.investment_widget.setLayout(investment_vbox)
        self.investment_widget.hide()

        self.share_price_label = QLabel()
        self.share_price_label.setObjectName("addStockLabel")
        self.share_price_label.setStyleSheet("color: black;")
        self.share_price_label.hide()

        investment_hbox = QHBoxLayout()

        self.investment_label = QLabel("Enter the amount of shares:")
        self.investment_label.setObjectName("addStockLabel")
        self.investment_label.hide()

        self.investment = QLineEdit()
        self.investment.setFixedSize(30, 30)
        self.investment.textChanged.connect(self.hide_invalid_investment_label)
        self.investment.returnPressed.connect(self.investment_entered)
        self.investment.hide()

        investment_hbox.addStretch(1)
        investment_hbox.setSpacing(10)
        investment_hbox.addWidget(self.investment_label)
        investment_hbox.addWidget(self.investment)
        investment_hbox.addStretch(1)

        self.invalid_investment_label = QLabel("The amount of shares has to be a positive integer.")
        self.invalid_investment_label.hide()
        self.invalid_investment_label.setStyleSheet("font-size: 15px; color: #C70C0C;")
        self.ticker_name.textChanged.connect(self.hide_invalid_investment_label)

        long_short_layout = QHBoxLayout()
        self.investment_long = QRadioButton("Long")
        self.investment_long.setObjectName("addStockLabel")
        self.investment_long.setChecked(True)
        self.investment_long.toggled.connect(self.sound_radio.play)
        self.investment_long.hide()
        self.investment_short = QRadioButton("Short")
        self.investment_short.setObjectName("addStockLabel")
        self.investment_short.toggled.connect(self.sound_radio.play)
        self.investment_short.hide()
        self.long_short_group = QButtonGroup(self)
        self.long_short_group.addButton(self.investment_long)
        self.long_short_group.addButton(self.investment_short)

        long_short_layout.setSpacing(30)
        long_short_layout.addStretch(1)
        long_short_layout.addWidget(self.investment_long)
        long_short_layout.addWidget(self.investment_short)
        long_short_layout.addStretch(1)

        buttons_hbox = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.add_button.setFixedSize(70, 40)
        self.add_button.setStyleSheet(button_save_style)
        self.add_button.clicked.connect(self.add_ticker_to_portfolio)

        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedSize(70, 40)
        cancel_button.clicked.connect(self.process_cancel_decision)

        buttons_hbox.addStretch(1)
        buttons_hbox.addWidget(cancel_button)
        spacer = QSpacerItem(30, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        buttons_hbox.addItem(spacer)
        buttons_hbox.addWidget(self.add_button)
        buttons_hbox.addStretch(1)

        ticker_vbox.addLayout(search_by_hbox)
        ticker_vbox.addWidget(ticker_label, alignment=Qt.AlignCenter)
        ticker_vbox.addLayout(ticker_hbox)
        ticker_vbox.addWidget(self.invalid_ticker_label, alignment=Qt.AlignCenter)

        investment_vbox.addWidget(self.share_price_label, alignment=Qt.AlignCenter)
        investment_vbox.addLayout(long_short_layout)
        investment_vbox.addLayout(investment_hbox)
        investment_vbox.addWidget(self.invalid_investment_label, alignment=Qt.AlignCenter)

        layout.addWidget(ticker_widget)
        layout.addWidget(self.investment_widget)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def search_by_radio_toggled(self, checked):
        """
        Method called when selected search list is changes
        :param checked: boolean
        """
        if checked:
            if self.by_name_radio.isChecked():
                self.current_list = self.sp500_companies
                newCompleterModel = QStringListModel(self.sp500_companies)
                self.completer.setModel(newCompleterModel)
            elif self.by_esg_radio.isChecked():
                self.current_list = self.top_100_esg_companies
                newCompleterModel = QStringListModel(self.top_100_esg_companies)
                self.completer.setModel(newCompleterModel)

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path

    def validate_ticker(self):
        """
        Method that validates a chosen stock to add
        """
        self.sound_radio.play()

        ticker = self.ticker_name.text()

        try:
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
                self.investment_widget.show()
            else:
                if price == -2:
                    self.invalid_ticker_label.setText("This stock is already part of your portfolio")
                else:
                    self.invalid_ticker_label.setText("Please choose a company from list")
                self.invalid_ticker_label.show()
        except:
            traceback.print_exc()
            self.main_window.show_error_window("Unable to read data from the API.",
                                               "Please check your Internet connection.")

    def is_valid_ticker(self, ticker):
        """
        Method that validates a chosen stock to add
        :param ticker: the chosen stock
        :return: True or False
        """
        if ticker.split()[0] in self.controller.tickers_and_investments.keys():
            return False, -2
        if (ticker in self.sp500_companies) or (ticker in self.top_100_esg_companies):
            ticker = ticker.split()[0]
            data = yf.Ticker(ticker)
            one_day_data = data.history(period="1d")
            if len(one_day_data) > 0:
                return True, one_day_data["Close"]
        else:
            return False, -1

    def validate_investment(self):
        """
        Method that validates a chosen number of shares
        """
        if not self.is_valid_investment(self.investment.text()):
            self.invalid_investment_label.show()
            self.investment.setFocus()

    def is_valid_investment(self, inv):
        """
        Method that validates a chosen number of shares
        :param ticker: the chosen number of shares
        :return: True or False
        """
        return bool(re.match(r'^[1-9]\d*$', inv))

    def add_ticker_to_portfolio(self):
        """
        Method that closes the pop-up and calls required methods to properly save the stock in portfolio
        """
        self.sound_action.play()
        num_shares = self.investment.text()
        if self.is_valid_investment(num_shares):
            if self.investment_long.isChecked():
                is_long = True
            else:
                is_long = False
            QTimer.singleShot(1000, lambda: self.valid_ticker_entered.emit(self.ticker.upper(), self.share_price,
                                                                           int(num_shares), is_long))
            QTimer.singleShot(1000, self.close)
            # self.valid_ticker_entered.emit(self.ticker.upper(), self.share_price, int(num_shares), is_long)
            # self.close()
        else:
            self.invalid_investment_label.show()
            self.investment.setFocus()

    def process_cancel_decision(self):
        """
        Method for closing the pop-up
        """
        self.sound_cancel.play()
        QTimer.singleShot(100, self.close)

    def hide_invalid_labels(self):
        """
        Method that hides labels saying "Invalid"
        """
        self.invalid_ticker_label.hide()
        self.invalid_investment_label.hide()

        self.share_price_label.hide()
        self.investment_long.hide()
        self.investment_short.hide()
        self.investment_label.hide()
        self.investment.hide()
        self.add_button.hide()
        self.investment_widget.hide()

    def hide_invalid_investment_label(self):
        """
        Method that hides a label saying "Invalid investment"
        """
        self.invalid_investment_label.hide()

    def investment_entered(self):
        """
        Method that sets focus on appropriate button
        """
        self.add_button.setFocus()

    def keyPressEvent(self, event):
        """
        Method called when a keyboard key is pressed
        :param event: key event
        """
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.accept()
            self.add_button.setFocus()
        else:
            super().keyPressEvent(event)


class CustomLineEdit(QLineEdit):
    """
    Custom line editor allowing only certain inpits
    """
    enter_pressed = pyqtSignal()

    def __init__(self, completer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completer = completer

    def keyPressEvent(self, event):
        """
        Method called when a keyboard key is pressed
        :param event: key event
        """
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
    """
    Pop-up for editing stock info
    """
    perform_change = pyqtSignal(str, int, float, bool)

    def __init__(self, ticker, stock_name, one_share_price, num_shares, is_long):
        """
        Constructor method
        :param ticker: ticker of stock to edit
        :param stock_name: name of stock
        :param one_share_price: price of 1 share
        :param num_shares: number of shares previously stored
        :param is_long: long/short investment
        """
        super().__init__()
        self.setWindowTitle(f"Edit {ticker} shares")

        self.ticker = ticker
        self.one_share_price = one_share_price
        self.is_long = is_long

        # Read the stylesheet file:
        path = self.get_file_path("style.css")
        with open(path, "r") as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        button_save_style = (
            "QPushButton:hover {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #12CB0B, stop:1 #0EC1D0);"
            "}"
            "QPushButton:pressed {"
            " background-color: black;"
            "}")
        button_delete_style = (
            "QPushButton:hover {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF0000, stop:1 #FF8600);"
            "}"
            "QPushButton:pressed {"
            " background-color: black;"
            "}")

        # Set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')
        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')
        radio_sound_path = os.path.join(sound_directory, 'radio_button.wav')

        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()
        self.sound_radio = QSoundEffect()

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))
        self.sound_radio.setSource(QUrl.fromLocalFile(radio_sound_path))

        # Build the page:

        layout = QVBoxLayout()

        ticker_vbox = QVBoxLayout()
        ticker_widget = QWidget()
        ticker_widget.setObjectName("addStockVBox")
        ticker_widget.setFixedSize(500, 200)
        ticker_widget.setLayout(ticker_vbox)

        self.stock_name_label = QLabel(f"{stock_name} ({ticker})")
        self.stock_name_label.setObjectName("addStockLabel")
        bold_font = QFont()
        bold_font.setBold(True)
        self.stock_name_label.setFont(bold_font)
        self.share_price_label = QLabel(f"Price of 1 share = ${one_share_price}")
        self.share_price_label.setObjectName("addStockLabel")

        investment_hbox = QHBoxLayout()

        self.investment_label = QLabel("Number of shares in portfolio:")
        self.investment_label.setObjectName("addStockLabel")

        self.investment = QLineEdit(str(num_shares))
        self.investment.setFixedSize(30, 30)
        self.investment.textChanged.connect(self.hide_invalid_investment_label)

        investment_hbox.addStretch(1)
        investment_hbox.addWidget(self.investment_label)
        investment_hbox.addWidget(self.investment)
        investment_hbox.addStretch(1)

        self.invalid_investment_label = QLabel("The amount of shares has to be a positive integer.")
        self.invalid_investment_label.hide()
        self.invalid_investment_label.setStyleSheet("font-size: 15px; color: #C70C0C;")
        self.invalid_investment_label.setStyleSheet("color: red;")

        long_short_layout = QHBoxLayout()
        long_short_layout.setSpacing(30)
        self.investment_long = QRadioButton("Long")
        self.investment_long.setObjectName("addStockLabel")
        self.investment_short = QRadioButton("Short")
        self.investment_short.setObjectName("addStockLabel")
        if is_long:
            self.investment_long.setChecked(True)
        else:
            self.investment_short.setChecked(True)
        self.investment_long.toggled.connect(self.sound_radio.play)
        self.investment_long.toggled.connect(self.long_short_changed)
        self.investment_short.toggled.connect(self.sound_radio.play)
        self.investment_short.toggled.connect(self.long_short_changed)
        long_short_layout.addStretch(1)
        long_short_layout.addWidget(self.investment_long)
        long_short_layout.addWidget(self.investment_short)
        long_short_layout.addStretch(1)

        buttons_hbox = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedSize(70, 40)
        cancel_button.clicked.connect(self.process_cancel_decision)

        self.save_changes_button = QPushButton("Save Changes")
        self.save_changes_button.setFixedSize(110, 40)
        self.save_changes_button.setStyleSheet(button_save_style)
        self.save_changes_button.clicked.connect(self.save_changes)
        self.save_changes_button.hide()

        self.delete_button = QPushButton(f"Delete {ticker} from portfolio")
        self.delete_button.setFixedSize(200, 40)
        self.delete_button.setStyleSheet(button_delete_style)
        self.delete_button.clicked.connect(self.delete_stock)

        buttons_hbox.addStretch(1)
        buttons_hbox.addWidget(cancel_button)
        spacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        buttons_hbox.addItem(spacer)
        buttons_hbox.addWidget(self.save_changes_button)
        spacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        buttons_hbox.addItem(spacer)
        buttons_hbox.addWidget(self.delete_button)
        buttons_hbox.addStretch(1)

        ticker_vbox.addWidget(self.stock_name_label, alignment=Qt.AlignCenter)
        ticker_vbox.addWidget(self.share_price_label, alignment=Qt.AlignCenter)
        ticker_vbox.addLayout(investment_hbox)
        ticker_vbox.addWidget(self.invalid_investment_label, alignment=Qt.AlignCenter)
        ticker_vbox.addLayout(long_short_layout)

        layout.addWidget(ticker_widget)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def process_cancel_decision(self):
        """
        Method for closing the pop-up
        """
        self.sound_cancel.play()
        QTimer.singleShot(100, self.close)

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path

    def hide_invalid_investment_label(self):
        """
        Method that hides labels saying "Invalid"
        """
        self.invalid_investment_label.hide()
        self.save_changes_button.show()

    def long_short_changed(self, checked):
        """
        Method called when user changes investment type
        :param checked: boolean
        """
        if checked:
            if self.investment_long.isChecked():
                self.is_long = True
            else:
                self.is_long = False
            self.save_changes_button.show()

    def save_changes(self):
        """
        Method for closing the pop-up and saving user requested changes
        """
        self.sound_action.play()
        num_shares = self.investment.text()
        if bool(re.match(r'^[1-9]\d*$', num_shares)):
            QTimer.singleShot(1000, lambda: self.perform_change.emit(self.ticker, int(num_shares),
                                                                     int(num_shares) * self.one_share_price,
                                                                     self.is_long))
            QTimer.singleShot(1000, self.close)
        else:
            self.invalid_investment_label.show()

    def delete_stock(self):
        """
        Method for closing the pop-up and deleting stock from portfolio
        """
        self.sound_action.play()
        QTimer.singleShot(1000, lambda: self.perform_change.emit(self.ticker, -1, -1, True))
        QTimer.singleShot(1000, self.close)

    def keyPressEvent(self, event):
        """
        Method called when keyboard key is pressed
        :param event: key event
        """
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.accept()
        else:
            super().keyPressEvent(event)


class RankingTimeWarningPopUp(QDialog):
    """
    Pop-up warning saying that algorithms ranking is timely
    """
    decision_made = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Warning")

        # Read the stylesheet file:
        path = self.get_file_path("style.css")
        with open(path, "r") as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        button_continue_style = (
            "QPushButton:hover {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF0000, stop:1 #FF8600);"
            "}"
            "QPushButton:pressed {"
            " background-color: black;"
            "}")

        # Set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')
        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')

        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))

        # Build the page:

        layout = QVBoxLayout()

        warning_vbox = QVBoxLayout()
        warning_vbox.setSpacing(30)
        warning_widget = QWidget()
        warning_widget.setObjectName("addStockVBox")
        warning_widget.setFixedSize(500, 120)
        warning_widget.setLayout(warning_vbox)

        warning_label = QLabel("The ranking process will take approximately 7 minutes.")
        warning_label.setObjectName("addStockLabel")
        warning_label_2 = QLabel("Do you still want to continue?")
        warning_label_2.setObjectName("addStockLabel")

        buttons_hbox = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedSize(70, 40)
        cancel_button.clicked.connect(self.sound_cancel.play)
        cancel_button.clicked.connect(self.process_cancel_decision)
        buttons_hbox.addWidget(cancel_button)

        continue_button = QPushButton("Continue")
        continue_button.setFixedSize(70, 40)
        continue_button.setStyleSheet(button_continue_style)
        continue_button.clicked.connect(self.sound_action.play)
        continue_button.clicked.connect(self.process_continue_decision)
        buttons_hbox.addWidget(continue_button)

        warning_vbox.addWidget(warning_label, alignment=Qt.AlignCenter)
        warning_vbox.addWidget(warning_label_2, alignment=Qt.AlignCenter)
        layout.addWidget(warning_widget)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path

    def process_cancel_decision(self):
        """
        Method for closing the pop-up
        """
        self.sound_cancel.play()
        QTimer.singleShot(100, lambda: self.decision_made.emit(False))
        QTimer.singleShot(100, self.close)

    def process_continue_decision(self):
        """
        Method for closing the pop-up and executing algorithms re-ranking
        """
        self.sound_action.play()
        QTimer.singleShot(1000, lambda: self.decision_made.emit(True))  # Delay for 100ms
        QTimer.singleShot(1000, self.close)
