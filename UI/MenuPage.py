import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

from UI.MainWindow import MainWindow
import logging


class MenuPage(QWidget):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.layout = QVBoxLayout()
        self.init_page()

        self.add_menu_buttons()

        self.setLayout(self.layout)

    def init_page(self):
        self.logger.info('Initializing the Main Menu Page')

        self.setWindowTitle('Main Menu')

        title = QLabel('Portfolio Analysis and Simulation Tool')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 30))
        title.setStyleSheet("color: red;")
        shadow_effect = QGraphicsDropShadowEffect(blurRadius=5, xOffset=3, yOffset=3)
        title.setGraphicsEffect(shadow_effect)
        self.layout.addWidget(title)

        self.layout.addStretch()

    def add_menu_buttons(self):
        portfolio_button = self.create_menu_button('Portfolio Analysis')
        manual_button = self.create_menu_button('User Manual')
        settings_button = self.create_menu_button('Settings')
        exit_button = self.create_menu_button('Exit')

        portfolio_button.clicked.connect(self.open_portfolio_page)
        manual_button.clicked.connect(self.open_manual_page)
        settings_button.clicked.connect(self.open_settings_page)
        exit_button.clicked.connect(self.quit_app)

        self.layout.addWidget(portfolio_button)
        self.layout.addWidget(manual_button)
        self.layout.addWidget(settings_button)
        self.layout.addWidget(exit_button)

        self.layout.addStretch()

    def create_menu_button(self, button_text):
        button = QPushButton(button_text)
        button.setFont(QFont('Arial', 16))
        button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 lightblue, stop:1 blue);
                border-style: outset;
                border-width: 2px;
                border-radius: 15px;
                border-color: beige;
                min-width: 200px;
                padding: 6px;
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 blue, stop:1 darkblue);
                border-style: inset;
            }
        """)
        return button

    def open_portfolio_page(self):
        self.logger.info('Opening the Portfolio Page')
        pass

    def open_manual_page(self):
        self.logger.info('Opening the User Manual Page')
        pass

    def open_settings_page(self):
        self.logger.info('Opening the Settings Page')
        pass

    def quit_app(self):
        self.logger.info('Exiting the Application')
        QApplication.instance().quit()
