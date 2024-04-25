import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication

from UI.MainWindow import MainWindow

# from Controller import Controller
# controller = Controller()
# controller.add_ticker("AAPL", 1, None, True)
# controller.handle_ranking(True)
# controller.run_arima("AAPL", "1w", True)


app = QApplication(sys.argv)

dpi = app.primaryScreen().logicalDotsPerInch()
main_window = MainWindow(dpi)

main_window.show()
sys.exit(app.exec_())

