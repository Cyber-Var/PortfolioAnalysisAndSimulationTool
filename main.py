import sys

from PyQt5.QtWidgets import QApplication

from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage


# controller = Controller("1m")
# controller.add_ticker("MSFT", 1000, True)
# controller.tune_hyperparameters("MSFT", 10000)

app = QApplication(sys.argv)

controller_1d = Controller("1w")
controller_1w = Controller("1w")
controller_1m = Controller("1m")
controllers = [controller_1d, controller_1w, controller_1m]

main_window = MainWindow()
menu_page = MenuPage(main_window, controllers)
main_window.setCentralWidget(menu_page)

main_window.show()
sys.exit(app.exec_())
