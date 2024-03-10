import sys

from PyQt5.QtWidgets import QApplication

from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage


controller = Controller()
controller.add_ticker("MSFT", 1000, True)
controller.tune_hyperparameters("MSFT", 10000, "1d")


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
