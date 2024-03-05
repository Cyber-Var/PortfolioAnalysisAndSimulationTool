import sys

from PyQt5.QtWidgets import QApplication

from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage


controller = Controller("1d")
controller.add_ticker("AAPL", 1000)
controller.tune_hyperparameters("AAPL", 10000)

# app = QApplication(sys.argv)
#
# controller = Controller("1w")
# main_window = MainWindow()
#
# menu_page = MenuPage(main_window, controller)
# main_window.setCentralWidget(menu_page)
#
# main_window.show()
# sys.exit(app.exec_())
