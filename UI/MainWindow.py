import sys

import pandas as pd
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator, QValidator
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QStackedWidget, QAction, QDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QComboBox, QLineEdit, QScrollArea, QWidget

from Controller import Controller
from UI.MenuPage import MenuPage
from UI.PortfolioPage import PortfolioPage
from UI.SingleStockPage import SingleStockPage


class MainWindow(QMainWindow):
    def __init__(self, dpi):
        super().__init__()

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

        self.menu_page = MenuPage(self, self.controller)
        self.portfolio_page = PortfolioPage(self, self.controller)
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

        # TODO: save activity option
        save_activity_option = QAction('Save Activity', self)
        save_activity_option.triggered.connect(self.close)
        options_menu.addAction(save_activity_option)

        exit_option = QAction('Exit the Application', self)
        exit_option.triggered.connect(self.close)
        options_menu.addAction(exit_option)

    def changePage(self, page):
        currentIndex = self.stacked_widget.currentIndex()
        currentPage = self.stacked_widget.widget(currentIndex)
        self.page_history.append(currentPage)

        self.stacked_widget.setCurrentWidget(page)

    def goBack(self):
        if self.page_history:
            previousPage = self.page_history.pop()
            self.stacked_widget.setCurrentWidget(previousPage)

    def openSingleStockPage(self, ticker, stock_name, one_share_price, num_shares, hold_duration):
        self.single_stock_page.set_parameters(ticker, stock_name, one_share_price, num_shares, hold_duration)
        self.changePage(self.single_stock_page)

    def show_top_esg(self):
        popup = TopESGPopUp()
        popup.exec_()


class TopESGPopUp(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Top ESG Companies")

        self.previous_top_N = 0
        self.top_N = 5
        self.top_companies_df = pd.read_csv('top_esg_companies.csv', sep=';')

        layout = QVBoxLayout()

        top_N_hbox = QHBoxLayout()
        layout.addLayout(top_N_hbox)

        top_label = QLabel("Top")
        top_label.setObjectName("inputLabel")
        top_N_hbox.addWidget(top_label)

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
        top_N_hbox.addWidget(self.top_N_combo)

        esg_label = QLabel("companies with highest ESG score:")
        esg_label.setObjectName("inputLabel")
        top_N_hbox.addWidget(esg_label)

        max_N_label = QLabel("(max 50)")
        max_N_label.setObjectName("inputLabel")
        layout.addWidget(max_N_label)

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

        top_companies = self.top_companies_df.head(self.top_N)

        # while self.top_companies_vbox.count():
        #     child = self.top_companies_vbox.takeAt(0)
        #     if child.widget():
        #         child.widget().deleteLater()

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
