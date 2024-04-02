import sys

from PyQt5.QtWidgets import QApplication

# from Controller import Controller
from UI.MainWindow import MainWindow
from UI.MenuPage import MenuPage
from UI.SingleStockPage import SingleStockPage

# controller = Controller()
# controller.handle_ranking()


app = QApplication(sys.argv)

dpi = app.primaryScreen().logicalDotsPerInch()
main_window = MainWindow(dpi)

main_window.show()
sys.exit(app.exec_())
