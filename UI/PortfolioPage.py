from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox,
                             QScrollArea, QDialog, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
import yfinance as yf

from UI.Page import Page


class PortfolioPage(QWidget, Page):

    def __init__(self, main_window, controller):
        super().__init__()

        self.main_window = main_window
        self.controller = controller

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

        self.ticker_col_name = None
        self.stock_name_col_name = None
        self.lin_reg_col_name = None
        self.random_forest_col_name = None
        self.bayesian_col_name = None
        self.monte_carlo_col_name = None
        self.arima_col_name = None
        self.lstm_col_name = None
        self.volatility_col_name = None
        self.more_info_col_name = None

        self.results_vbox = None
        self.results_map = {}

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Portfolio Analysis")

        self.build_page()

        self.setStyleSheet(self.load_stylesheet())

        self.setLayout(self.layout)

    def build_page(self):
        title_label = self.get_title_label("Portfolio Analysis and Simulation Tool")

        input_hbox = QHBoxLayout()

        self.hold_duration_label = self.get_title_label("Hold Duration:")
        self.hold_duration_label.setObjectName('inputLabel')
        input_hbox.addWidget(self.hold_duration_label)

        hold_duration_vbox = QVBoxLayout()
        input_hbox.addLayout(hold_duration_vbox)

        self.hold_duration_1d = self.create_hold_duration_button("1 day")
        self.hold_duration_1d.setChecked(True)
        self.hold_duration_1w = self.create_hold_duration_button("1 week")
        self.hold_duration_1m = self.create_hold_duration_button("1 month")

        hold_duration_vbox.addWidget(self.hold_duration_1d)
        hold_duration_vbox.addWidget(self.hold_duration_1w)
        hold_duration_vbox.addWidget(self.hold_duration_1m)

        self.algorithms_label = self.get_title_label("Algorithms:")
        self.algorithms_label.setObjectName('inputLabel')
        input_hbox.addWidget(self.algorithms_label)

        algorithms_vbox = QVBoxLayout()
        input_hbox.addLayout(algorithms_vbox)

        self.algorithm_1 = self.create_algorithm_checkbox("Linear Regression", 0)
        self.algorithm_2 = self.create_algorithm_checkbox("Random Forest", 1)
        self.algorithm_3 = self.create_algorithm_checkbox("Bayesian", 2)
        self.algorithm_4 = self.create_algorithm_checkbox("Monte Carlo", 3)
        self.algorithm_5 = self.create_algorithm_checkbox("ARIMA", 4)
        self.algorithm_6 = self.create_algorithm_checkbox("LSTM", 5)

        algorithms_vbox.addWidget(self.algorithm_1)
        algorithms_vbox.addWidget(self.algorithm_2)
        algorithms_vbox.addWidget(self.algorithm_3)
        algorithms_vbox.addWidget(self.algorithm_4)
        algorithms_vbox.addWidget(self.algorithm_5)
        algorithms_vbox.addWidget(self.algorithm_6)

        self.results_vbox = QVBoxLayout()

        column_names_hbox = QHBoxLayout()
        # results_vbox.addLayout(column_names_hbox)

        self.ticker_col_name = self.create_column_names_labels("Ticker")
        self.stock_name_col_name = self.create_column_names_labels("Name")
        self.lin_reg_col_name = self.create_column_names_labels("Linear Regression")
        self.lin_reg_col_name.hide()
        self.random_forest_col_name = self.create_column_names_labels("Random Forest")
        self.random_forest_col_name.hide()
        self.bayesian_col_name = self.create_column_names_labels("Bayesian")
        self.bayesian_col_name.hide()
        self.monte_carlo_col_name = self.create_column_names_labels("Monte Carlo")
        self.monte_carlo_col_name.hide()
        self.arima_col_name = self.create_column_names_labels("ARIMA")
        self.arima_col_name.hide()
        self.lstm_col_name = self.create_column_names_labels("LSTM")
        self.lstm_col_name.hide()
        self.volatility_col_name = self.create_column_names_labels("Volatility")
        self.more_info_col_name = self.create_column_names_labels("More Info")

        column_names_hbox.addWidget(self.ticker_col_name)
        column_names_hbox.addWidget(self.stock_name_col_name)
        column_names_hbox.addWidget(self.lin_reg_col_name)
        column_names_hbox.addWidget(self.random_forest_col_name)
        column_names_hbox.addWidget(self.bayesian_col_name)
        column_names_hbox.addWidget(self.monte_carlo_col_name)
        column_names_hbox.addWidget(self.arima_col_name)
        column_names_hbox.addWidget(self.lstm_col_name)
        column_names_hbox.addWidget(self.volatility_col_name)
        column_names_hbox.addWidget(self.more_info_col_name)

        add_stock_button = QPushButton("+ Add Stock")
        add_stock_button.setObjectName('addStockButton')
        add_stock_button.clicked.connect(self.show_add_stock_window)
        # self.results_vbox.addWidget(add_stock_button)

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

        self.layout.addWidget(title_label)
        self.layout.addLayout(input_hbox)
        self.layout.addWidget(scrollable_area)

    def create_hold_duration_button(self, name):
        button = QRadioButton(name)
        button.setObjectName('inputLabel')
        button.toggled.connect(self.hold_duration_button_toggled)
        return button

    def create_algorithm_checkbox(self, name, ii):
        button = QCheckBox(name)
        button.setObjectName('inputLabel')
        button.stateChanged.connect(lambda state, index=ii: self.algorithms_state_changed(state, index))
        return button

    def create_column_names_labels(self, name):
        label = QLabel(name)
        label.setObjectName('columnNameLabel')
        label.setFixedHeight(50)
        return label

    def hold_duration_button_toggled(self):
        # if self.hold_duration_1d.isChecked():
        #     self.hold_duration_label.setText("Selected option: 1 day")
        # elif self.hold_duration_1w.isChecked():
        #     self.hold_duration_label.setText("Selected option: 1 week")
        # elif self.hold_duration_1m.isChecked():
        #     self.hold_duration_label.setText("Selected option: 1 month")
        pass

    def algorithms_state_changed(self, state, index):
        if state == Qt.Checked:
            self.algorithms[index] = True
        else:
            self.algorithms[index] = False

    def show_add_stock_window(self):
        popup = AddStockPopUp()
        popup.valid_ticker_entered.connect(self.add_ticker)
        popup.exec_()

    def add_ticker(self, ticker, investment):
        self.controller.add_ticker(ticker, investment)

        results_hbox = QHBoxLayout()

        ticker_label = QLabel(ticker)
        ticker_label.setObjectName("resultLabel")
        ticker_label.setFixedHeight(50)
        results_hbox.addWidget(ticker_label)

        stock_info = yf.Ticker(ticker).info
        stock_name = stock_info.get('longName', 'N/A')
        stock_name_label = QLabel(stock_name)
        stock_name_label.setObjectName("resultLabel")
        stock_name_label.setFixedHeight(50)
        results_hbox.addWidget(stock_name_label)

        if self.algorithms[0]:
            self.lin_reg_col_name.show()
            lin_reg_prediction = self.controller.run_linear_regression(ticker)
            lin_reg_label = QLabel(f"{lin_reg_prediction[0][0]:.2f}")
            lin_reg_label.setObjectName("resultLabel")
            lin_reg_label.setFixedHeight(50)
            results_hbox.addWidget(lin_reg_label)
        if self.algorithms[1]:
            self.random_forest_col_name.show()
            random_forest_prediction = self.controller.run_random_forest(ticker)
            random_forest_label = QLabel(f"{random_forest_prediction[0][0]:.2f}")
            random_forest_label.setObjectName("resultLabel")
            random_forest_label.setFixedHeight(50)
            results_hbox.addWidget(random_forest_label)
        if self.algorithms[2]:
            self.bayesian_col_name.show()
            bayesian_prediction = self.controller.run_bayesian(ticker)
            bayesian_label = QLabel(f"{bayesian_prediction[0][0]:.2f}")
            bayesian_label.setObjectName("resultLabel")
            bayesian_label.setFixedHeight(50)
            results_hbox.addWidget(bayesian_label)
        if self.algorithms[3]:
            self.monte_carlo_col_name.show()
            # TODO: num_of_simulations set by user
            monte_carlo_prediction = self.controller.run_monte_carlo(ticker)
            monte_carlo_prediction_label = QLabel(monte_carlo_prediction)
            monte_carlo_prediction_label.setObjectName("resultLabel")
            monte_carlo_prediction_label.setFixedHeight(50)
            results_hbox.addWidget(monte_carlo_prediction_label)
        if self.algorithms[4]:
            self.arima_col_name.show()
            arima_prediction = self.controller.run_arima(ticker)
            arima_prediction_label = QLabel(f"{arima_prediction.iloc[-1]:.2f}")
            arima_prediction_label.setObjectName("resultLabel")
            arima_prediction_label.setFixedHeight(50)
            results_hbox.addWidget(arima_prediction_label)
        if self.algorithms[5]:
            self.lstm_col_name.show()
            lstm_prediction = self.controller.run_lstm(ticker)
            lstm_prediction_label = QLabel(f"{lstm_prediction[0][0]:.2f}")
            lstm_prediction_label.setObjectName("resultLabel")
            lstm_prediction_label.setFixedHeight(50)
            results_hbox.addWidget(lstm_prediction_label)

        volatility, category = self.controller.get_volatility(ticker)
        volatility_label = QLabel(f"{volatility:.2f} {category}")
        volatility_label.setObjectName("resultLabel")
        volatility_label.setFixedHeight(50)
        results_hbox.addWidget(volatility_label)

        more_info_button = QPushButton("--->")
        more_info_button.clicked.connect((lambda state, ticker_name=ticker: self.show_single_share_page(ticker_name)))
        more_info_button.setObjectName("moreInfoButton")
        more_info_button.setFixedHeight(50)
        results_hbox.addWidget(more_info_button)

        self.results_vbox.addLayout(results_hbox)
        self.results_map[ticker] = results_hbox

    def show_single_share_page(self, ticker_name):
        print(f"Show {ticker_name} page")
        pass



class AddStockPopUp(QDialog):

    valid_ticker_entered = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Stock")

        layout = QVBoxLayout()

        ticker_label = QLabel("Please enter the stock ticker to add:")
        self.ticker_name = QLineEdit()

        self.invalid_label = QLabel("Invalid")
        self.invalid_label.hide()
        self.invalid_label.setStyleSheet("color: red;")
        self.ticker_name.textChanged.connect(self.hide_invalid_label)

        investment_label = QLabel("and the investment amount:")
        self.investment = QLineEdit()

        buttons_hbox = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.validate_ticker)
        buttons_hbox.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        buttons_hbox.addWidget(cancel_button)

        layout.addWidget(ticker_label)
        layout.addWidget(self.ticker_name)
        layout.addWidget(investment_label)
        layout.addWidget(self.investment)
        layout.addWidget(self.invalid_label)
        layout.addLayout(buttons_hbox)

        self.setLayout(layout)

    def validate_ticker(self):
        ticker = self.ticker_name.text()
        investment = self.investment.text()
        if self.is_valid(ticker):
            self.valid_ticker_entered.emit(ticker, investment)
            self.close()
        else:
            self.invalid_label.show()

    def is_valid(self, ticker):
        try:
            data = yf.Ticker(ticker)
            one_day_data = data.history(period="1d")
            if len(one_day_data) > 0:
                return True
            return False
        except Exception:
            return False

    def hide_invalid_label(self):
        self.invalid_label.hide()
