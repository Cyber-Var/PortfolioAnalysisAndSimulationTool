import sys

from PyQt5.QtWidgets import QApplication

from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage
from UI.SingleStockPage import SingleStockPage

# from Controller import Controller
# controller = Controller()


app = QApplication(sys.argv)

dpi = app.primaryScreen().logicalDotsPerInch()
main_window = MainWindow(dpi)

main_window.show()
sys.exit(app.exec_())
