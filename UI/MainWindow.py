import sys
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QStackedWidget

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
        # Add other pages here

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.portfolio_page)
        self.stacked_widget.addWidget(self.single_stock_page)
        # Add widgets to the stack

        self.menu_page.open_portfolio_page.connect(lambda: self.changePage(self.portfolio_page))
        self.portfolio_page.back_to_menu_page.connect(self.goBack)
        self.portfolio_page.open_single_stock_page.connect(self.openSingleStockPage)
        self.single_stock_page.back_to_portfolio_page.connect(self.goBack)

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
