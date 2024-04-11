import logging
import os

from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont, QPalette, QColor


class Page:

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.load_stylesheet()

        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()
        self.sound_radio = QSoundEffect()

    def init_page(self, page_name):
        self.logger.info(f'Initializing the {page_name} Page')
        self.setWindowTitle(page_name)

        sound_directory = os.path.join(os.path.dirname(__file__), 'sounds')

        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')
        radio_sound_path = os.path.join(sound_directory, 'radio_button.wav')

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))
        self.sound_radio.setSource(QUrl.fromLocalFile(radio_sound_path))

    def play_action_sound(self):
        self.sound_action.play()

    def play_cancel_sound(self):
        self.sound_cancel.play()

    def play_radio_sound(self):
        self.sound_radio.play()

    def get_title_label(self, text, style_name):
        title_label = QLabel(text)
        title_label.setObjectName(style_name)
        title_label.setAlignment(Qt.AlignCenter)
        return title_label

    def load_stylesheet(self):
        with open("UI/style.css", "r") as f:
            stylesheet = f.read()
        return stylesheet
