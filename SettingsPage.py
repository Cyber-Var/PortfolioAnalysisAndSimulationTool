import subprocess

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QHBoxLayout, QRadioButton
from PyQt5.QtCore import Qt, pyqtSignal

from Page import Page


class SettingsPage(QWidget, Page):
    """
    Settings page with settings for sound volume and frequency of re-ranking algorithms
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

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Settings")

        self.volume_changer = None
        self.every_day_radio = None
        self.every_week_radio = None
        self.every_month_radio = None

        self.build_page()

        self.setLayout(self.layout)

    def set_ranking_frequency(self, freq):
        """
        Setter for the re-ranking frequency
        :param freq: new frequency
        """
        self.ranking_frequency = freq

    def build_page(self):
        """
        Method for building the page
        :return:
        """
        self.logger.info('Opening the Settings Menu Page')

        self.layout.setSpacing(50)

        title_label = self.get_title_label("Settings", "titleLabelSettings")
        self.layout.addWidget(title_label)

        volume_hbox = QHBoxLayout()
        volume_hbox.setSpacing(10)
        volume_widget = QWidget()
        volume_widget.setObjectName("inputHBox")
        volume_widget.setFixedSize(500, 200)
        volume_widget.setLayout(volume_hbox)

        volume_label = QLabel("Sound Volume:")
        volume_label.setObjectName("volumeLabel")
        volume_label.setFixedSize(200, 30)

        self.volume_changer = QSlider(Qt.Horizontal, self)
        self.volume_changer.setMinimum(0)
        self.volume_changer.setMaximum(100)
        self.volume_changer.setValue(100)
        self.volume_changer.setSingleStep(1)
        self.volume_changer.setPageStep(10)
        self.volume_changer.setFixedSize(200, 30)
        self.volume_changer.valueChanged.connect(self.change_volume)

        ranking_frequency_hbox = QHBoxLayout()
        ranking_frequency_hbox.setSpacing(10)
        ranking_frequency_widget = QWidget()
        ranking_frequency_widget.setObjectName("inputHBox")
        ranking_frequency_widget.setFixedSize(500, 200)
        ranking_frequency_widget.setLayout(ranking_frequency_hbox)

        ranking_frequency_label = QLabel("Frequency of algorithms ranking:")
        ranking_frequency_label.setObjectName("inputHeaderLabel")
        ranking_frequency_label.setFixedSize(320, 30)

        ranking_frequency_vbox = QVBoxLayout()

        self.every_day_radio = self.create_ranking_frequency_radio_button("every day")
        self.every_week_radio = self.create_ranking_frequency_radio_button("every week")
        self.every_month_radio = self.create_ranking_frequency_radio_button("every month")
        self.set_checked()

        ranking_frequency_vbox.addWidget(self.every_day_radio)
        ranking_frequency_vbox.addWidget(self.every_week_radio)
        ranking_frequency_vbox.addWidget(self.every_month_radio)

        ranking_frequency_hbox.addWidget(ranking_frequency_label)
        ranking_frequency_hbox.addLayout(ranking_frequency_vbox)

        back_button = QPushButton("Back")
        back_button.setObjectName("addStockButton")
        back_button.setFixedSize(90, 40)
        back_button.clicked.connect(self.play_cancel_sound)
        back_button.clicked.connect(self.back_to_menu_page.emit)

        volume_hbox.addWidget(volume_label)
        volume_hbox.addWidget(self.volume_changer)

        self.layout.addWidget(volume_widget, alignment=Qt.AlignCenter)
        self.layout.addWidget(ranking_frequency_widget, alignment=Qt.AlignCenter)

        self.layout.addStretch()
        self.layout.addWidget(back_button)

    def change_volume(self, volume):
        """
        Method for changing volume
        :param volume: new volume level
        """
        script = f'set volume output volume {volume}'
        subprocess.run(["osascript", "-e", script])

    def create_ranking_frequency_radio_button(self, name):
        """
        Method for creating one ranking frequency radio button
        :param name: name of the button
        :return: the radio button
        """
        button = QRadioButton(name)
        button.setObjectName('inputLabel')
        button.toggled.connect(self.play_radio_sound)
        button.toggled.connect(self.ranking_frequency_button_toggled)
        return button

    def set_checked(self):
        """
        Set the correct ranking frequency radio button to checked state
        """
        if self.controller.ranking_frequency == 0:
            self.every_day_radio.setChecked(True)
        elif self.controller.ranking_frequency == 1:
            self.every_week_radio.setChecked(True)
        else:
            self.every_month_radio.setChecked(True)

    def ranking_frequency_button_toggled(self, checked):
        """
        Method called when a user selects a new ranking frequency
        :return:
        """
        if checked:
            self.logger.info('Handling the change in ranking frequency.')
            if self.every_day_radio.isChecked():
                self.ranking_frequency = 0
                self.main_window.update_ranking_frequency(0)
            elif self.every_week_radio.isChecked():
                self.ranking_frequency = 1
                self.main_window.update_ranking_frequency(1)
            elif self.every_month_radio.isChecked():
                self.ranking_frequency = 2
                self.main_window.update_ranking_frequency(2)
