import logging
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QGraphicsDropShadowEffect, \
    QStackedLayout, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox, QScrollArea, QDialog, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import yfinance as yf


from UI.MainWindow import MainWindow
from UI.Page import Page


class PortfolioPage(QWidget, Page):

    def __init__(self, main_window):
        super().__init__()

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

        self.ticker_col_name = None
        self.stock_name_col_name = None
        self.volatility_col_name = None
        self.more_info_col_name = None

        self.main_window = main_window

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

        self.algorithm_1 = self.create_algorithm_checkbox("Linear Regression")
        self.algorithm_2 = self.create_algorithm_checkbox("Random Forest")
        self.algorithm_3 = self.create_algorithm_checkbox("Bayesian")
        self.algorithm_4 = self.create_algorithm_checkbox("Monte Carlo")
        self.algorithm_5 = self.create_algorithm_checkbox("ARIMA")
        self.algorithm_6 = self.create_algorithm_checkbox("LSTM")

        algorithms_vbox.addWidget(self.algorithm_1)
        algorithms_vbox.addWidget(self.algorithm_2)
        algorithms_vbox.addWidget(self.algorithm_3)
        algorithms_vbox.addWidget(self.algorithm_4)
        algorithms_vbox.addWidget(self.algorithm_5)
        algorithms_vbox.addWidget(self.algorithm_6)

        results_vbox = QVBoxLayout()

        column_names_hbox = QHBoxLayout()
        # results_vbox.addLayout(column_names_hbox)

        self.ticker_col_name = self.create_column_names_labels("Ticker")
        self.stock_name_col_name = self.create_column_names_labels("Name")
        self.volatility_col_name = self.create_column_names_labels("Volatility")
        self.more_info_col_name = self.create_column_names_labels("More Info")

        column_names_hbox.addWidget(self.ticker_col_name)
        column_names_hbox.addWidget(self.stock_name_col_name)
        column_names_hbox.addWidget(self.volatility_col_name)
        column_names_hbox.addWidget(self.more_info_col_name)

        add_stock_button = QPushButton("+ Add Stock")
        add_stock_button.setObjectName('addStockButton')
        add_stock_button.clicked.connect(self.add_stock)
        results_vbox.addWidget(add_stock_button)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("border: 2px solid white;")
        scroll_layout = QVBoxLayout()
        scroll_layout.addLayout(column_names_hbox)
        scroll_layout.addStretch(1)
        scroll_layout.addLayout(results_vbox)
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        self.layout.addWidget(title_label)
        self.layout.addLayout(input_hbox)
        self.layout.addWidget(scroll_area)

    def create_hold_duration_button(self, name):
        button = QRadioButton(name)
        button.setObjectName('inputLabel')
        button.toggled.connect(self.hold_duration_button_toggled)
        return button

    def create_algorithm_checkbox(self, name):
        button = QCheckBox(name)
        button.setObjectName('inputLabel')
        button.stateChanged.connect(self.algorithms_state_changed)
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

    def algorithms_state_changed(self):
        pass

    def add_stock(self):
        popup = AddStockPopUp()
        popup.exec_()


class AddStockPopUp(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Stock")

        layout = QVBoxLayout()

        label = QLabel("Please enter the stock ticker to add:")
        self.ticker_name = QLineEdit()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.validate_ticker)

        layout.addWidget(label)
        layout.addWidget(self.ticker_name)
        layout.addWidget(ok_button)

        self.setLayout(layout)

    def validate_ticker(self):
        ticker = self.ticker_name.text()
        if self.is_valid(ticker):
            print("Valid")
        else:
            print("Not Valid")

    def is_valid(self, ticker):
        try:
            data = yf.Ticker(ticker)
            one_day_data = data.history(period="1d")
            if len(one_day_data) > 0:
                return True
            return False
        except Exception:
            return False
