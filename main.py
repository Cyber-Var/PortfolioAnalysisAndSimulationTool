import sys

from PyQt5.QtWidgets import QApplication

from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage


controller = Controller()
controller.add_ticker("AAPL", 1000, True)
controller.add_ticker("TSLA", 2000, False)
controller.add_ticker("NVDA", 3000, True)
controller.calculate_risk_metrics("AAPL", "1d")
controller.calculate_risk_metrics("TSLA", "1d")
controller.calculate_risk_metrics("NVDA", "1d")


# app = QApplication(sys.argv)
#
# controller = Controller()
#
# main_window = MainWindow()
# menu_page = MenuPage(main_window, controller)
# main_window.setCentralWidget(menu_page)
#
# main_window.show()
# sys.exit(app.exec_())
