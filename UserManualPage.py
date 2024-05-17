import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal

from Page import Page


class UserManualPage(QWidget, Page):
    """
    Page providing user guide and explanation of the tool's algorithms
    """

    back_to_menu_page = pyqtSignal()

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

        self.current_index = -1
        self.is_guide = None

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("User Manual")

        self.title_label = None
        self.image_label = None
        self.guide_button = None
        self.algorithms_button = None

        self.cwd = os.getcwd()

        self.build_page()

        self.setLayout(self.layout)

    def build_page(self):
        """
        Method for building the page
        """
        self.layout = QVBoxLayout()
        self.logger.info('Opening the User Manual Page')

        self.layout.setSpacing(25)

        self.title_label = self.get_title_label("User Manual", "titleLabelManual")
        self.layout.addWidget(self.title_label)

        self.add_menu_buttons()

        self.image_label = QLabel()
        self.image_label.hide()
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.video_links_vbox = QVBoxLayout()
        self.video_links_widget = QWidget()
        self.video_links_widget.setObjectName("inputHBox")
        self.video_links_widget.setFixedSize(800, 100)
        self.video_links_widget.setLayout(self.video_links_vbox)
        self.video_links_widget.hide()

        self.layout.addWidget(self.video_links_widget, alignment=Qt.AlignCenter)

        buttons_hbox = QHBoxLayout()

        self.back_to_manual_button = QPushButton("Exit Guide")
        self.back_to_manual_button.setObjectName("addStockButton")
        self.back_to_manual_button.setFixedSize(90, 40)
        self.back_to_manual_button.hide()
        self.back_to_manual_button.clicked.connect(self.play_cancel_sound)
        self.back_to_manual_button.clicked.connect(self.exit_guide)

        back_button = QPushButton("Back")
        back_button.setObjectName("addStockButton")
        back_button.setFixedSize(90, 40)
        back_button.clicked.connect(self.play_cancel_sound)
        back_button.clicked.connect(self.go_back)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("addStockButton")
        self.next_button.setFixedSize(90, 40)
        self.next_button.hide()
        self.next_button.clicked.connect(self.play_cancel_sound)
        self.next_button.clicked.connect(self.go_next)

        buttons_hbox.addStretch()
        buttons_hbox.addWidget(self.back_to_manual_button)
        buttons_hbox.addWidget(back_button)
        buttons_hbox.addWidget(self.next_button)
        buttons_hbox.addStretch()

        self.layout.addStretch(1)
        self.layout.addLayout(buttons_hbox)

    def add_menu_buttons(self):
        """
        Method for displaying menu buttons
        """
        self.guide_button = self.create_menu_button('User Guide')
        self.algorithms_button = self.create_menu_button('Explanation of Algorithms')

        self.guide_button.clicked.connect(self.open_guide)
        self.algorithms_button.clicked.connect(self.open_algorithms)

        self.layout.addStretch()

    def add_video_link(self, text, link):
        """
        Method for adding links to related tutorials to the display
        :param text: text before link
        :param link: the link
        """
        label = QLabel(f"{text}: {link}")
        label.setStyleSheet("color: white; font-size: 15px;")
        label.setTextInteractionFlags(label.textInteractionFlags() | Qt.TextSelectableByMouse)
        self.video_links_vbox.addWidget(label)

    def open_guide(self):
        """
        Method for opening the user guide
        """
        self.is_guide = True
        self.go_next()

    def open_algorithms(self):
        """
        Method for opening explanations of algorithms
        """
        self.is_guide = False
        self.go_next()

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

    def show_init(self):
        """
        Method for rendering initial screen:
        """
        self.guide_button.show()
        self.algorithms_button.show()
        self.image_label.hide()
        self.next_button.hide()

    def show_guide(self):
        """
        Method for showing a page with one guide / algorithmic explanation
        """

        # Show user guide:
        if self.is_guide:
            self.image_label.setFixedSize(1100, 650)
            if self.current_index == 0:
                self.title_label.setText("Portfolio Analysis Page")
            elif self.current_index == 1:
                self.title_label.setText("Single Stock Page")
            elif self.current_index == 2:
                self.title_label.setText("Settings Page")
            elif self.current_index == 3:
                self.title_label.setText("Top Companies by ESG Score Page")

            image_path = self.get_file_path(f"guide_{self.current_index}.png")
            pixmap = QPixmap(image_path)
        # Show explanation of an algorithm:
        else:
            self.image_label.setFixedSize(1100, 500)
            for i in reversed(range(self.video_links_vbox.count())):
                widget = self.video_links_vbox.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            self.video_links_widget.show()

            if self.current_index == 0:
                self.title_label.setText("Linear Regression")
                self.add_video_link("IBM's explanation", "https://youtu.be/qxo8p8PtFeA?si=S8bAhJghaU-BEz0A")
                self.add_video_link("<Visually Explained>'s explanation", "https://youtu.be/CtsRRUddV2s?si=d6Mnd8lUC8WAyvM7")
            elif self.current_index == 1:
                self.title_label.setText("Random Forest Regression")
                self.add_video_link("IBM's explanation", "https://youtu.be/gkXX4h3qYm4?si=AohLdfmQqe8gnGK4")
                self.add_video_link("<Super Data Science>'s explanation", "https://youtu.be/X1MRbEnEq2s?si=564CwNREWiOoQ3wc")
            elif self.current_index == 2:
                self.title_label.setText("Bayesian Ridge Regression")
                self.add_video_link("<ritvikmath>", "https://youtu.be/Z6HGJMUakmc?si=zc1y7uR6W1NSeFJ2")
                self.add_video_link("<Geostats Lectures>' explanation", "https://youtu.be/LzZ5b3wdZQk?si=UxsgUH5KTzQBfecY")
            elif self.current_index == 3:
                self.title_label.setText("Monte Carlo Simulation")
                self.add_video_link("IBM's explanation", "https://youtu.be/7TqhmX92P6U?si=t_Zm3B9zD5OIQFUH")
                self.add_video_link("<365 Financial Analyst>'s explanation", "https://youtu.be/hhBNk0xmZ9U?si=nFxiq5e_QdqWuJ-M")
            elif self.current_index == 4:
                self.title_label.setText("ARIMA")
                self.add_video_link("DecisionForest's explanation", "https://youtu.be/gqryqIlvEoM?si=pu2LRn2WY38zPWCz")
                self.add_video_link("Intellipaat's explanation", "https://youtu.be/4Fiz3dQM_i8?si=p4e7JmTWIlLuJFBA")
            elif self.current_index == 5:
                self.title_label.setText("Risk Metrics")
                self.add_video_link("Volatility", "https://youtu.be/3_jjS3x3oC0?si=igev8DofX8IT8NYK")
                self.add_video_link("Sharpe ratio", "https://youtu.be/QpVhm_Ab84k?si=Cz_u5ZGoXFKTOJd7")
                self.add_video_link("VaR", "https://youtu.be/2SMkbMDypXI?si=V5_lmRybYIdiAkZP")
            elif self.current_index == 6:
                self.title_label.setText("ESG Scores")
                self.add_video_link("<Bloomberg Law>'s explanation", "https://youtu.be/-WVdP9ssU2o?si=_ZFcgPE7r2CMnJOH")
                self.add_video_link("<Corporate Finance Institute>'s explanation", "https://youtu.be/AkbGz3CYvqE?si=nLY9jjkgqrQAlDDG")

            image_path = self.get_file_path(f"algorithm_{self.current_index}.png")
            pixmap = QPixmap(image_path)

        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.show()

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path

    def go_next(self):
        """
        Method for opening the next part of the guide:
        """
        self.current_index += 1
        self.guide_button.hide()
        self.algorithms_button.hide()
        if self.is_guide:
            if self.current_index == 3:
                self.next_button.hide()
            else:
                self.next_button.show()
        else:
            if self.current_index == 6:
                self.next_button.hide()
            else:
                self.next_button.show()
        if self.current_index >= 1:
            self.back_to_manual_button.show()
        self.show_guide()

    def go_back(self):
        """
        Method for opening the next previous of the guide:
        """
        self.current_index -= 1
        if self.current_index < 1:
            self.back_to_manual_button.hide()
        if self.current_index == -2:
            self.back_to_menu_page.emit()
        elif self.current_index == -1:
            self.next_button.hide()
            self.video_links_widget.hide()
            self.title_label.setText("User Manual")
            self.image_label.hide()
            self.guide_button.show()
            self.algorithms_button.show()
        else:
            self.next_button.show()
            self.show_guide()

    def exit_guide(self):
        """
        Method for exiting a guide
        """
        self.current_index = 0
        self.go_back()


