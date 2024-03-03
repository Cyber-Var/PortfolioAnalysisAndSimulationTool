import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: black;")

        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(frame_geom.topLeft())

