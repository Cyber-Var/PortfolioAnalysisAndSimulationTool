from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from Page import Page


class MenuPage(QWidget, Page):
    """
    Menu page with navigation around the app
    """

    open_portfolio_page = pyqtSignal()
    open_settings_page = pyqtSignal()
    open_manual_page = pyqtSignal()

    def __init__(self, main_window, controller):
        """
        Constructor method
        :param main_window: main window of the app
        :param controller: the controller object
        """
        super().__init__()

        self.main_window = main_window
        self.controller = controller

        # Set up the page:

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Main Menu")

        self.build_page()

        self.setLayout(self.layout)

    def build_page(self):
        """
        Method for building the page
        """
        self.logger.info('Opening the Main Menu Page')

        self.layout.setSpacing(25)

        title_label = self.get_title_label("Portfolio Analysis and Simulation Tool", "titleLabelMenu")
        self.layout.addWidget(title_label)

        self.add_menu_buttons()
        self.layout.addStretch(1)

    def add_menu_buttons(self):
        """
        Method for displaying the menu buttons
        """
        portfolio_button = self.create_menu_button('Portfolio Analysis')
        manual_button = self.create_menu_button('User Manual')
        settings_button = self.create_menu_button('Settings')
        exit_button = self.create_menu_button('Exit')

        portfolio_button.clicked.connect(self.open_portfolio_page.emit)
        manual_button.clicked.connect(self.open_manual_page)
        settings_button.clicked.connect(self.open_settings_page.emit)
        exit_button.clicked.connect(self.quit_app)

        self.layout.addStretch()

    def create_menu_button(self, name):
        """
        Method for creating one menu button
        :param name: name of the button
        :return: the button
        """
        button = QPushButton(name)
        button.setObjectName('menuButton')
        button.setFixedWidth(400)
        button.clicked.connect(self.play_cancel_sound)
        self.layout.addWidget(button, alignment=Qt.AlignHCenter)
        return button

    def quit_app(self):
        """
        Method for closig the application
        """
        self.logger.info('Exiting the Application')
        self.play_cancel_sound()
        QTimer.singleShot(500, lambda: self.main_window.closeEvent(None))
        QTimer.singleShot(500, QApplication.instance().quit)
