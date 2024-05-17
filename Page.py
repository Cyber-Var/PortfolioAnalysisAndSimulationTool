import logging
import os

from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont, QPalette, QColor


class Page:
    """
    Class with settings used by each page type within the application
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()

        # Set up the Logger and style file:
        self.logger = logging.getLogger(__name__)
        self.load_stylesheet()

        # Initialize sound variables:
        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()
        self.sound_radio = QSoundEffect()

    def init_page(self, page_name):
        """
        Method for initializing a page
        :param page_name: name of the page to display on page title
        """

        # Set the name of page's title:
        self.logger.info(f'Initializing the {page_name} Page')
        self.setWindowTitle(page_name)

        # Set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')

        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')
        radio_sound_path = os.path.join(sound_directory, 'radio_button.wav')

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))
        self.sound_radio.setSource(QUrl.fromLocalFile(radio_sound_path))

    def play_action_sound(self):
        """
        Method for playing the sound of action
        """
        self.sound_action.play()

    def play_cancel_sound(self):
        """
        Method for playing the sound of cancelling/navigation
        """
        self.sound_cancel.play()

    def play_radio_sound(self):
        """
        Method for playing the sound of a radio button
        """
        self.sound_radio.play()

    def get_title_label(self, text, style_name):
        """
        Method for creating the title label
        :param text: name of the label
        :param style_name: CSS style of the label
        :return: the title label
        """
        title_label = QLabel(text)
        title_label.setObjectName(style_name)
        title_label.setAlignment(Qt.AlignCenter)
        return title_label

    def load_stylesheet(self):
        """
        Method for loading the application's stylesheet from file:
        :return: the stylesheet
        """
        path = self.get_file_path("style.css")
        with open(path, 'r') as file:
            stylesheet = file.read()
        return stylesheet

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path
