from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

import logging

from UI.Page import Page
from UI.PortfolioPage import PortfolioPage


class MenuPage(QWidget, Page):

    open_portfolio_page = pyqtSignal()

    def __init__(self, main_window, controller):
        super().__init__()

        self.main_window = main_window
        self.controller = controller

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Main Menu")

        self.build_page()

        self.setLayout(self.layout)

    def build_page(self):
        self.logger.info('Opening the Main Menu Page')

        self.layout.setSpacing(25)

        title_label = self.get_title_label("Portfolio Analysis and Simulation Tool", "titleLabelMenu")
        self.layout.addWidget(title_label)

        self.add_menu_buttons()
        self.layout.addStretch(1)

    def add_menu_buttons(self):
        portfolio_button = self.create_menu_button('Portfolio Analysis')
        manual_button = self.create_menu_button('User Manual')
        settings_button = self.create_menu_button('Settings')
        exit_button = self.create_menu_button('Exit')

        portfolio_button.clicked.connect(self.open_portfolio_page.emit)
        manual_button.clicked.connect(self.open_manual_page)
        settings_button.clicked.connect(self.open_settings_page)
        exit_button.clicked.connect(self.quit_app)

        self.layout.addStretch()

    def create_menu_button(self, name):
        button = QPushButton(name)
        button.setObjectName('menuButton')
        button.setFixedWidth(400)
        self.layout.addWidget(button, alignment=Qt.AlignHCenter)
        return button

    def open_manual_page(self):
        self.logger.info('Opening the User Manual Page')
        pass

    def open_settings_page(self):
        self.logger.info('Opening the Settings Page')
        pass

    def quit_app(self):
        self.logger.info('Exiting the Application')
        QApplication.instance().quit()
