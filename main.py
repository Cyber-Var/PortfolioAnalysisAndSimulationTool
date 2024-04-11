import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication

from UI.MainWindow import MainWindow

# from Controller import Controller
# controller = Controller()
# controller.add_ticker("AAPL", 2, None, True)
# controller.run_monte_carlo("AAPL", "1w")


app = QApplication(sys.argv)

dpi = app.primaryScreen().logicalDotsPerInch()
main_window = MainWindow(dpi)

main_window.show()
sys.exit(app.exec_())

