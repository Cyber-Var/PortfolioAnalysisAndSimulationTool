import sys

from PyQt5.QtWidgets import QApplication

from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage
from UI.SingleStockPage import SingleStockPage

# controller = Controller()
# controller.handle_ranking()


app = QApplication(sys.argv)

controller = Controller()

main_window = MainWindow()
menu_page = MenuPage(main_window, controller)
main_window.setCentralWidget(menu_page)

main_window.show()
sys.exit(app.exec_())

# app = QApplication(sys.argv)
#
# controller = Controller()
#
# main_window = MainWindow()
# single_stock_page = SingleStockPage(main_window, controller, "AAPL", "APPLE", 171.21,
#                                     2, "1d")
# main_window.setCentralWidget(single_stock_page)
#
# main_window.show()
# sys.exit(app.exec_())
