import sys

from PyQt5.QtWidgets import QApplication

from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage
from UI.SingleStockPage import SingleStockPage

# from Controller import Controller
# controller = Controller()
# controller.get_sp500_tickers()
# controller.add_ticker("AAPL", 1, None, True)
# controller.add_ticker("TSLA", 2, None, False)
# controller.add_ticker("MSFT", 3, None, True)
# print(controller.get_sharpe_ratio("AAPL", "1d"))
# print(controller.get_sharpe_ratio("TSLA", "1d"))
# print(controller.get_sharpe_ratio("MSFT", "1d"))


app = QApplication(sys.argv)

dpi = app.primaryScreen().logicalDotsPerInch()
main_window = MainWindow(dpi)

main_window.show()
sys.exit(app.exec_())
