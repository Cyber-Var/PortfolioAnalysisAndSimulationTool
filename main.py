import sys

from PyQt5.QtWidgets import QApplication

from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage


controller = Controller("1m")
controller.add_ticker("MSFT", 1000)
controller.tune_hyperparameters("MSFT", 10000)

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
