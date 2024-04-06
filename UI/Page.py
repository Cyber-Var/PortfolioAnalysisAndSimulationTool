import logging
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor


class Page:

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.load_stylesheet()

    def init_page(self, page_name):
        self.logger.info(f'Initializing the {page_name} Page')
        self.setWindowTitle(page_name)

    def get_title_label(self, text, style_name):
        title_label = QLabel(text)
        title_label.setObjectName(style_name)
        title_label.setAlignment(Qt.AlignCenter)
        return title_label

    def load_stylesheet(self):
        with open("UI/style.css", "r") as f:
            stylesheet = f.read()
        return stylesheet
