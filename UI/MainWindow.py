import logging
import os
import sys
import traceback
from datetime import date

import pandas as pd
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator, QValidator
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QStackedWidget, QAction, QDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QComboBox, QLineEdit, QScrollArea, QWidget, QMenu

from Controller import Controller
from UI.MenuPage import MenuPage
from UI.PortfolioPage import PortfolioPage
from UI.SingleStockPage import SingleStockPage


class MainWindow(QMainWindow):
    user_activity_file_name = "user_activity.txt"

    def __init__(self, dpi):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.dpi = dpi

        self.setGeometry(0, 0, 1300, 900)
        self.setStyleSheet("background-color: black;")

        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(frame_geom.topLeft())

        self.page_history = []

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.controller = Controller()
        self.hold_durations = ["1d", "1w", "1m"]
        set_algorithms, hold_duration = self.set_up_controller()

        self.menu_page = MenuPage(self, self.controller)
        self.portfolio_page = PortfolioPage(self, self.controller, set_algorithms, hold_duration)
        self.single_stock_page = SingleStockPage(self, self.controller, dpi)

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.portfolio_page)
        self.stacked_widget.addWidget(self.single_stock_page)

        self.menu_page.open_portfolio_page.connect(lambda: self.changePage(self.portfolio_page))
        self.portfolio_page.back_to_menu_page.connect(self.goBack)
        self.portfolio_page.open_single_stock_page.connect(self.openSingleStockPage)
        self.single_stock_page.back_to_portfolio_page.connect(self.goBack)

        options_menu = self.menuBar().addMenu('Options Menu')

        view_top_esg_option = QAction('View top ESG companies', self)
        view_top_esg_option.triggered.connect(self.show_top_esg)
        options_menu.addAction(view_top_esg_option)

        ranking_frequency_menu = QMenu('Update ranking every:', self)
        self.previous_frequency = 2

        daily_updates = QAction('day', self)
        daily_updates.triggered.connect(lambda: self.update_ranking_frequency(0))
        ranking_frequency_menu.addAction(daily_updates)

        weekly_updates = QAction('week', self)
        weekly_updates.triggered.connect(lambda: self.update_ranking_frequency(1))
        ranking_frequency_menu.addAction(weekly_updates)

        monthly_updates = QAction('month', self)
        monthly_updates.triggered.connect(lambda: self.update_ranking_frequency(2))
        ranking_frequency_menu.addAction(monthly_updates)

        update_ranking_frequency_option = QAction('Frequency of algorithms ranking', self)
        update_ranking_frequency_option.setMenu(ranking_frequency_menu)
        options_menu.addAction(update_ranking_frequency_option)

        save_activity_option = QAction('Save Activity', self)
        save_activity_option.triggered.connect(self.closeEvent)
        options_menu.addAction(save_activity_option)

    def set_up_controller(self):
        self.logger.info(f'Reading previously saved user activity from file')
        set_algorithms = [False, False, False, False, False, False]
        try:
            with (open(self.user_activity_file_name, "r") as f):
                last_date = f.readline().strip()
                if last_date != date.today().strftime("%d.%m.%Y"):
                    should_update = True
                else:
                    should_update = False

                hold_duration = f.readline().strip()
                while True:
                    lines = [f.readline() for _ in range(4)]

                    if all(not line for line in lines):
                        break

                    one_stock_data = lines[0].strip().split("|")
                    ticker = one_stock_data[0]
                    if one_stock_data[2] == "True":
                        is_long = True
                    else:
                        is_long = False
                    self.controller.add_ticker(ticker, int(one_stock_data[1]), None, is_long)

                    alg_results_1d = lines[1].strip().split("|")[:6]
                    alg_results_1w = lines[2].strip().split("|")[:6]
                    alg_results_1m = lines[3].strip().split("|")[:6]
                    alg_results = [alg_results_1d, alg_results_1w, alg_results_1m]

                    for i in range(len(self.hold_durations)):
                        hold_dur = self.hold_durations[i]
                        for index in range(len(alg_results[i])):
                            if alg_results[i][index] != "":
                                set_algorithms[index] = True
                                if index == 3 or index == 5 or should_update:
                                    self.controller.run_algorithm(ticker, index, hold_dur)
                                else:
                                    alg_name = self.controller.algorithms_with_indices[index]
                                    res = alg_results[i][index].split(",")
                                    self.controller.results[alg_name][hold_dur][ticker] = float(res[0])
                                    self.controller.predicted_prices[alg_name][hold_dur][ticker] = float(res[1])
                                    if index == 2:
                                        self.controller.bayesian_confidences[hold_dur][ticker] = (float(res[2]), float(res[3]))

            return set_algorithms, hold_duration
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"User activity failed to be read. {e}")
            return set_algorithms, "1d"

    def changePage(self, page):
        currentIndex = self.stacked_widget.currentIndex()
        currentPage = self.stacked_widget.widget(currentIndex)
        self.page_history.append(currentPage)

        self.stacked_widget.setCurrentWidget(page)

    def goBack(self):
        if self.page_history:
            previousPage = self.page_history.pop()
            self.stacked_widget.setCurrentWidget(previousPage)

    def openSingleStockPage(self, ticker, stock_name, one_share_price, num_shares, hold_duration, is_long):
        self.single_stock_page.set_parameters(ticker, stock_name, one_share_price, num_shares, hold_duration, is_long)
        self.changePage(self.single_stock_page)

    def show_top_esg(self):
        popup = TopESGPopUp()
        popup.exec_()

    def update_ranking_frequency(self, freq):
        if self.previous_frequency != freq:
            self.controller.ranking_frequency = freq
            self.controller.handle_ranking()
        self.previous_frequency = freq

    def closeEvent(self, event):
        self.logger.info(f'Saving user activity to file.')
        try:
            with (open(self.user_activity_file_name, 'w') as f):
                today = date.today().strftime("%d.%m.%Y")
                f.write(f"{today}\n")
                f.write(f"{self.portfolio_page.hold_duration}\n")
                for ticker in self.controller.tickers_and_investments.keys():
                    f.write(f"{ticker}|{self.controller.tickers_and_num_shares[ticker]}|"
                            f"{self.controller.tickers_and_long_or_short[ticker]}\n")

                    for hold_dur in self.hold_durations:
                        str_result = ""
                        for algorithm in self.controller.results:
                            alg_results = self.controller.results[algorithm]
                            alg_predicted_prices = self.controller.predicted_prices[algorithm]

                            if ticker in alg_results[hold_dur]:
                                if algorithm == "monte_carlo" or algorithm == "arima":
                                    str_result += "a|"
                                else:
                                    str_result += str(alg_results[hold_dur][ticker]) + "," + str(alg_predicted_prices[hold_dur][ticker]) + "|"
                                    if algorithm == "bayesian":
                                        confidence = self.controller.bayesian_confidences[hold_dur][ticker]
                                        str_result = str_result[:-1] + "," + str(confidence[0]) + "," + str(confidence[1]) + "|"
                            else:
                                str_result += "|"
                        f.write(f"{str_result}\n")
        except IOError:
            self.logger.error("Saving user activity to file failed.")


class TopESGPopUp(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Top companies by ESG score")

        with open("UI/style.css", "r") as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        button_save_style = (
            "QPushButton:hover {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #12CB0B, stop:1 #0EC1D0);"
            "}"
            "QPushButton:pressed {"
            " background-color: black;"
            "}")

        self.logger = logging.getLogger(__name__)
        self.logger.info("Displaying the Top ESG companies page")

        self.previous_top_N = 0
        self.top_N = 5
        self.top_companies_df = pd.read_csv('top_esg_companies.csv', sep=';')

        layout = QVBoxLayout()

        top_label = QLabel("Top")
        top_label.setObjectName("addStockLabel")

        self.top_N_combo = CustomComboBoxESG()
        self.top_N_combo.setFixedWidth(50)
        self.top_N_combo.setEditable(True)
        self.top_N_combo.addItem('5')
        self.top_N_combo.addItem('10')
        self.top_N_combo.addItem('20')
        self.top_N_combo.addItem('50')
        self.top_N_combo.setCurrentIndex(0)
        self.top_N_combo.lineEdit().setPlaceholderText("5")
        self.top_N_combo.activated.connect(self.top_N_changed)

        esg_label = QLabel("companies with highest ESG score:")
        esg_label.setObjectName("addStockLabel")

        top_N_hbox = QHBoxLayout()
        top_N_hbox.setSpacing(10)
        top_N_hbox.addStretch(1)
        top_N_hbox.addWidget(top_label)
        top_N_hbox.addWidget(self.top_N_combo)
        top_N_hbox.addWidget(esg_label)
        top_N_hbox.addStretch(1)

        max_N_label = QLabel("(max 50)")
        max_N_label.setObjectName("topESGLabel")

        self.place_col_name = self.create_column_names_label("Place")
        self.ticker_col_name = self.create_column_names_label("Ticker")
        self.stock_name_col_name = self.create_column_names_label("Stock Name")
        self.industry_col_name = self.create_column_names_label("Industry")
        self.column_names_hbox = self.display_column_names()

        self.top_companies_vbox = QVBoxLayout()
        self.scrollable_area = QScrollArea()
        self.scrollable_area.setWidgetResizable(True)
        self.scrollable_widget = QWidget()
        self.scrollable_widget.setStyleSheet("border: 2px solid white;")
        scrollable_layout = QVBoxLayout()
        scrollable_layout.addLayout(self.column_names_hbox)
        scrollable_layout.addLayout(self.top_companies_vbox)
        scrollable_layout.addStretch(1)
        self.scrollable_widget.setLayout(scrollable_layout)
        self.scrollable_area.setWidget(self.scrollable_widget)

        layout.addLayout(top_N_hbox)
        layout.addWidget(max_N_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.scrollable_area)

        self.display_top_esg_companies(False)
        self.setLayout(layout)

    def display_column_names(self):
        column_names_hbox = QHBoxLayout()

        column_names_hbox.addWidget(self.place_col_name)
        column_names_hbox.addWidget(self.ticker_col_name)
        column_names_hbox.addWidget(self.stock_name_col_name)
        column_names_hbox.addWidget(self.industry_col_name)

        return column_names_hbox

    def create_column_names_label(self, name):
        label = QLabel(name)
        label.setObjectName('columnNameLabel')
        label.setFixedHeight(50)
        return label

    def create_result_label(self, name):
        label = QLabel(name)
        label.setObjectName('resultLabel')
        label.setFixedHeight(50)
        return label

    def display_top_esg_companies(self, not_first=True):
        if self.top_N == self.previous_top_N and not_first:
            return

        self.logger.info(f"Displaying top {self.top_N} ESG companies.")

        top_companies = self.top_companies_df.head(self.top_N)

        self.column_names_hbox = self.display_column_names()
        new_scrollable_widget = QWidget()
        new_scrollable_widget.setStyleSheet("border: 2px solid white;")
        self.top_companies_vbox = QVBoxLayout()
        self.top_companies_vbox.addLayout(self.column_names_hbox)
        new_scrollable_widget.setLayout(self.top_companies_vbox)
        self.scrollable_area.setWidget(new_scrollable_widget)

        for i in range(self.top_N):
            results_hbox = QHBoxLayout()
            index_label = self.create_result_label(str(top_companies["index"].iloc[i]))
            ticker_label = self.create_result_label(top_companies["ticker"].iloc[i])
            stock_name_label = self.create_result_label(top_companies["stock_name"].iloc[i])
            industry_label = self.create_result_label(top_companies["industry"].iloc[i])

            results_hbox.addWidget(index_label)
            results_hbox.addWidget(ticker_label)
            results_hbox.addWidget(stock_name_label)
            results_hbox.addWidget(industry_label)

            self.top_companies_vbox.addLayout(results_hbox)

        self.top_companies_vbox.addStretch(1)

    def top_N_changed(self):
        top_N_entered = self.top_N_combo.currentText()
        if top_N_entered != "":
            self.previous_top_N = self.top_N
            self.top_N = int(top_N_entered)
            self.display_top_esg_companies()


class CustomComboBoxESG(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)

        validator = CustomValidator(1, 50, self)
        self.lineEdit().setValidator(validator)


class CustomValidator(QValidator):
    def __init__(self, minimum, maximum, parent=None):
        super().__init__(parent)
        self.min = minimum
        self.max = maximum

    def validate(self, inp, pos):
        if inp:
            try:
                value = int(inp)
                if self.min <= value <= self.max:
                    return QValidator.Acceptable, inp, pos
                else:
                    return QValidator.Invalid, inp, pos
            except ValueError:
                return QValidator.Invalid, inp, pos
        else:
            return QValidator.Intermediate, inp, pos

    def fixup(self, inp):
        try:
            value = int(inp)
            if value > self.max:
                return str(self.max)
            if value < self.min:
                return str(5)
        except ValueError:
            return str(self.min)
