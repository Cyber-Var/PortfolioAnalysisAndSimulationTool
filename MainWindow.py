import logging
import os
import traceback
from datetime import date

import pandas as pd
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QValidator
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QStackedWidget, QAction, QDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QComboBox, QScrollArea, QWidget, QMenu, QPushButton

from Controller import Controller
from MenuPage import MenuPage
from PortfolioPage import PortfolioPage
from SettingsPage import SettingsPage
from SingleStockPage import SingleStockPage
from UserManualPage import UserManualPage


class MainWindow(QMainWindow):
    """
    Main window of the application
    """

    user_activity_file_name = "user_activity.txt"

    def __init__(self, dpi):
        """
        Constructor method
        :param dpi: the user computer's dpi (in PPI)
        """
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.dpi = dpi

        # Set up the widget:
        self.setGeometry(0, 0, 1425, 900)
        self.setStyleSheet("background-color: black;")

        # Set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')
        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')

        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))

        # Center the window on the user's screen:

        self.frame_geom = self.frameGeometry()
        self.frame_geom.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(self.frame_geom.topLeft())

        # Set up navigation between pages:

        self.page_history = []

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Read top companies by ESG scores:

        path = self.get_file_path("top_esg_companies.csv")
        self.top_companies_df = pd.read_csv(path, sep=';')
        self.tickers_and_esg_indices = list(self.top_companies_df["ticker"])

        # Set up controller:
        self.controller = Controller()
        self.previous_frequency = self.controller.read_ranking_frequency()
        self.hold_durations = ["1d", "1w", "1m"]
        set_algorithms, hold_duration = self.set_up_controller()

        # Initialise pages:

        self.menu_page = MenuPage(self, self.controller)
        self.portfolio_page = PortfolioPage(self, self.controller, set_algorithms, hold_duration)
        self.single_stock_page = SingleStockPage(self, self.controller, dpi)
        self.settings_page = SettingsPage(self, self.controller)
        self.user_manual_page = UserManualPage(self, self.controller)

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.user_manual_page)
        self.stacked_widget.addWidget(self.portfolio_page)
        self.stacked_widget.addWidget(self.single_stock_page)

        self.menu_page.open_portfolio_page.connect(lambda: self.changePage(self.portfolio_page))
        self.menu_page.open_settings_page.connect(self.openSettingsPage)
        self.menu_page.open_manual_page.connect(self.openManualPage)
        self.portfolio_page.back_to_menu_page.connect(self.goBack)
        self.user_manual_page.back_to_menu_page.connect(self.goBack)
        self.portfolio_page.open_single_stock_page.connect(self.openSingleStockPage)
        self.single_stock_page.back_to_portfolio_page.connect(self.goBack)
        self.settings_page.back_to_menu_page.connect(self.goBack)

        # Set up the top menu bar:

        options_menu = self.menuBar().addMenu('Options Menu')

        view_top_esg_option = QAction('View top ESG companies', self)
        view_top_esg_option.triggered.connect(self.sound_cancel.play)
        view_top_esg_option.triggered.connect(self.show_top_esg)
        options_menu.addAction(view_top_esg_option)
        options_menu.addSeparator()

        ranking_frequency_menu = QMenu("", self)

        daily_updates = QAction('every day', self)
        daily_updates.triggered.connect(self.sound_action.play)
        daily_updates.triggered.connect(lambda: self.update_ranking_frequency(0))
        ranking_frequency_menu.addAction(daily_updates)
        ranking_frequency_menu.addSeparator()

        weekly_updates = QAction('every week', self)
        weekly_updates.triggered.connect(self.sound_action.play)
        weekly_updates.triggered.connect(lambda: self.update_ranking_frequency(1))
        ranking_frequency_menu.addAction(weekly_updates)
        ranking_frequency_menu.addSeparator()

        monthly_updates = QAction('every month', self)
        monthly_updates.triggered.connect(self.sound_action.play)
        monthly_updates.triggered.connect(lambda: self.update_ranking_frequency(2))
        ranking_frequency_menu.addAction(monthly_updates)
        ranking_frequency_menu.addSeparator()

        update_ranking_frequency_option = QAction('Frequency of algorithms ranking', self)
        update_ranking_frequency_option.triggered.connect(self.sound_cancel.play)
        update_ranking_frequency_option.setMenu(ranking_frequency_menu)
        options_menu.addAction(update_ranking_frequency_option)
        options_menu.addSeparator()

        save_activity_option = QAction('Save Activity', self)
        save_activity_option.triggered.connect(self.sound_action.play)
        save_activity_option.triggered.connect(self.closeEvent)
        options_menu.addAction(save_activity_option)
        options_menu.addSeparator()

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path

    def set_up_controller(self):
        """
        Method for reading previously saved user activity and setting up the controller with this data
        :return: algorithms chosen
        """
        self.logger.info(f'Reading previously saved user activity from file')
        set_algorithms = [False, False, False, False, False]
        try:
            # Read the file:
            path = self.get_file_path(self.user_activity_file_name)
            with open(path, "r") as f:
                last_date = f.readline().strip()
                if last_date != date.today().strftime("%d.%m.%Y"):
                    should_update = True
                else:
                    should_update = False

                # Loop through file and store each algorithmic result:
                hold_duration = f.readline().strip()
                while True:
                    lines = [f.readline() for _ in range(4)]

                    if all(not line for line in lines):
                        break

                    one_stock_data = lines[0].strip().split("|")
                    ticker = one_stock_data[0]
                    if one_stock_data[2] == "True":
                        is_long = True
                    else:
                        is_long = False
                    self.controller.add_ticker(ticker, int(one_stock_data[1]), None, is_long)

                    alg_results_1d = lines[1].strip().split("|")[:6]
                    alg_results_1w = lines[2].strip().split("|")[:6]
                    alg_results_1m = lines[3].strip().split("|")[:6]
                    alg_results = [alg_results_1d, alg_results_1w, alg_results_1m]

                    for i in range(len(self.hold_durations)):
                        hold_dur = self.hold_durations[i]
                        for index in range(len(alg_results[i])):
                            if alg_results[i][index] != "":
                                set_algorithms[index] = True
                                if index == 3 or should_update:
                                    self.controller.run_algorithm(ticker, index, hold_dur)
                                else:
                                    alg_name = self.controller.algorithms_with_indices[index]
                                    res = alg_results[i][index].split(",")
                                    self.controller.results[alg_name][hold_dur][ticker] = float(res[0])
                                    self.controller.predicted_prices[alg_name][hold_dur][ticker] = float(res[1])
                                    if index == 2:
                                        self.controller.bayesian_confidences[hold_dur][ticker] = (
                                        float(res[2]), float(res[3]))
                                    if index == 4:
                                        self.controller.arima_confidences[hold_dur][ticker] = (
                                        float(res[2]), float(res[3]))

            return set_algorithms, hold_duration
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"User activity failed to be read. {e}")
            self.show_error_window("Error occurred when loading the portfolio.", "Some stocks might not be displayed.",
                                   "Please check your Internet connection.")
            return set_algorithms, "1d"

    def changePage(self, page):
        """
        Method for opening a new page
        :param page: page to open
        """
        self.format_page_size(page)

        currentIndex = self.stacked_widget.currentIndex()
        currentPage = self.stacked_widget.widget(currentIndex)
        self.page_history.append(currentPage)

        self.stacked_widget.setCurrentWidget(page)

    def goBack(self):
        """
        Method for returning to the previous page
        """
        if self.page_history:
            previousPage = self.page_history.pop()
            self.format_page_size(previousPage)
            self.stacked_widget.setCurrentWidget(previousPage)

    def format_page_size(self, page):
        """
        Method for handling the page size
        :param page: page to format
        """
        if page == self.portfolio_page:
            self.setGeometry(0, 0, 1470, 900)
        else:
            self.setGeometry(0, 0, 1300, 900)
        self.frame_geom.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(self.frame_geom.topLeft())

    def openSingleStockPage(self, ticker, stock_name, one_share_price, num_shares, hold_duration, is_long):
        """
        Method for opening a Single Stock page
        :param ticker: stock for which to open page
        :param stock_name: name of the stock
        :param one_share_price: price of 1 stock's share
        :param num_shares: number of shares invested into
        :param hold_duration: user selected hold duration
        :param is_long: long/short investment
        """
        self.single_stock_page.set_parameters(ticker, stock_name, one_share_price, num_shares, hold_duration, is_long)
        self.changePage(self.single_stock_page)

    def openSettingsPage(self):
        """
        Method for opening the Settings page
        """
        self.settings_page.set_checked()
        self.changePage(self.settings_page)

    def openManualPage(self):
        """
        Method for opening the User Manual page
        """
        self.user_manual_page.current_index = -1
        self.user_manual_page.show_init()
        self.changePage(self.user_manual_page)

    def show_top_esg(self):
        """
        Method for opening the pop-up with top companies by ESG scores
        """
        popup = TopESGPopUp(self)
        popup.exec_()

    def update_ranking_frequency(self, freq):
        """
        Setter of the automatic ranking frequency
        :param freq: new frequency
        """
        self.logger.info(f"Updating ranking frequency to {freq}")
        if self.previous_frequency != freq:
            self.controller.set_ranking_frequency(freq)
            self.controller.handle_ranking()
        self.previous_frequency = freq

    def show_error_window(self, text_1, text_2, text_3=None):
        """
        Method for displaying a pop-up with error message
        :param text_1: first line of error text
        :param text_2: second line of error text
        :param text_3: third line of error text
        :return:
        """
        popup = ErrorPopUp(text_1, text_2, text_3)
        popup.exec_()

    def closeEvent(self, event):
        """
        Method executed when application is closed that saves user activity to file
        :param event: closing event
        """
        self.logger.info(f'Saving user activity to file.')
        try:
            # Open file for saving user activity:
            path = self.get_file_path(self.user_activity_file_name)
            with open(path, 'w') as f:
                today = date.today().strftime("%d.%m.%Y")
                f.write(f"{today}\n")
                f.write(f"{self.portfolio_page.hold_duration}\n")

                # Loop through and save all algorithmic results:
                for ticker in self.controller.tickers_and_investments.keys():
                    f.write(f"{ticker}|{self.controller.tickers_and_num_shares[ticker]}|"
                            f"{self.controller.tickers_and_long_or_short[ticker]}\n")

                    for hold_dur in self.hold_durations:
                        str_result = ""
                        for algorithm in self.controller.results:
                            alg_results = self.controller.results[algorithm]
                            alg_predicted_prices = self.controller.predicted_prices[algorithm]

                            if ticker in alg_results[hold_dur]:
                                if algorithm == "monte_carlo":
                                    str_result += "a|"
                                else:
                                    str_result += str(alg_results[hold_dur][ticker]) + "," + str(
                                        alg_predicted_prices[hold_dur][ticker]) + "|"
                                    if algorithm == "bayesian":
                                        confidence = self.controller.bayesian_confidences[hold_dur][ticker]
                                        str_result = str_result[:-1] + "," + str(confidence[0]) + "," + str(
                                            confidence[1]) + "|"
                                    elif algorithm == "arima":
                                        confidence = self.controller.arima_confidences[hold_dur][ticker]
                                        str_result = str_result[:-1] + "," + str(confidence[0]) + "," + str(
                                            confidence[1]) + "|"
                            else:
                                str_result += "|"
                        f.write(f"{str_result}\n")
        except IOError:
            self.logger.error("Saving user activity to file failed.")
            self.show_error_window(f"Error occurred when saving your activity.",
                                   "Some results might not be displayed on your next visit.")


class TopESGPopUp(QDialog):
    """
    Pop-up displaying top companies by ESG scores
    """

    def __init__(self, main_window):
        """
        Constructor method
        :param main_window: main window of the app
        """
        super().__init__()
        self.setWindowTitle("Top companies by ESG score")

        self.main_window = main_window

        # Read style file:
        path = self.get_file_path("style.css")
        with open(path, "r") as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)

        # set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')
        action_sound_path = os.path.join(sound_directory, 'action_button.wav')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')

        self.sound_cancel = QSoundEffect()
        self.sound_action = QSoundEffect()

        self.sound_action.setSource(QUrl.fromLocalFile(action_sound_path))
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))

        self.logger = logging.getLogger(__name__)
        self.logger.info("Displaying the Top ESG companies page")

        self.previous_top_N = 0
        self.top_N = 5

        # Build the page:

        layout = QVBoxLayout()

        top_label = QLabel("Top")
        top_label.setObjectName("addStockLabel")

        self.top_N_combo = CustomComboBoxESG()
        self.top_N_combo.setFixedWidth(50)
        self.top_N_combo.setEditable(True)
        self.top_N_combo.addItem('5')
        self.top_N_combo.addItem('10')
        self.top_N_combo.addItem('20')
        self.top_N_combo.addItem('50')
        self.top_N_combo.addItem('100')
        self.top_N_combo.setCurrentIndex(0)
        self.top_N_combo.lineEdit().setPlaceholderText("5")
        self.top_N_combo.activated.connect(self.top_N_changed)

        esg_label = QLabel("companies with lowest (best) ESG score:")
        esg_label.setObjectName("addStockLabel")

        top_N_hbox = QHBoxLayout()
        top_N_hbox.setSpacing(10)
        top_N_hbox.addStretch(1)
        top_N_hbox.addWidget(top_label)
        top_N_hbox.addWidget(self.top_N_combo)
        top_N_hbox.addWidget(esg_label)
        top_N_hbox.addStretch(1)

        max_N_label = QLabel("(max 100)")
        max_N_label.setObjectName("topESGLabel")
        max_N_label.setStyleSheet("font-size: 15px; font-weight: normal; margin-left: 170px;")

        self.place_col_name = self.create_column_names_label("Place")
        self.place_col_name.setFixedWidth(60)
        self.ticker_col_name = self.create_column_names_label("Ticker")
        self.ticker_col_name.setFixedWidth(60)
        self.stock_name_col_name = self.create_column_names_label("Stock Name")
        self.stock_name_col_name.setFixedWidth(200)
        self.industry_col_name = self.create_column_names_label("Industry")
        self.industry_col_name.setFixedWidth(350)
        self.column_names_hbox = self.display_column_names()

        self.top_companies_vbox = QVBoxLayout()
        self.scrollable_area = QScrollArea()
        self.scrollable_area.setWidgetResizable(True)
        self.scrollable_area.setFixedWidth(710)
        self.scrollable_widget = QWidget()
        self.scrollable_widget.setStyleSheet("border: 2px solid white;")
        scrollable_layout = QVBoxLayout()
        scrollable_layout.addLayout(self.column_names_hbox)
        scrollable_layout.addLayout(self.top_companies_vbox)
        scrollable_layout.addStretch(1)
        self.scrollable_widget.setLayout(scrollable_layout)
        self.scrollable_area.setWidget(self.scrollable_widget)

        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedSize(70, 40)
        cancel_button.clicked.connect(self.process_cancel_decision)

        layout.addLayout(top_N_hbox)
        layout.addWidget(max_N_label, alignment=Qt.AlignLeft)
        layout.addWidget(self.scrollable_area)
        layout.addWidget(cancel_button, alignment=Qt.AlignCenter)

        self.display_top_esg_companies(False)
        self.setLayout(layout)

        self.finished.connect(self.on_finished)

    def display_column_names(self):
        """
        Method that displays column names of the page
        :return: HBox with column names
        """
        column_names_hbox = QHBoxLayout()
        column_names_hbox.setSpacing(3)

        column_names_hbox.addWidget(self.place_col_name)
        column_names_hbox.addWidget(self.ticker_col_name)
        column_names_hbox.addWidget(self.stock_name_col_name)
        column_names_hbox.addWidget(self.industry_col_name)

        return column_names_hbox

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path

    def process_cancel_decision(self):
        """
        Method that closes the pop-up
        """
        self.sound_cancel.play()
        QTimer.singleShot(100, self.close)

    def create_column_names_label(self, name):
        """
        Method that creates one label used for column name
        :param name: name of the label
        :return: the label
        """
        label = QLabel(name)
        label.setObjectName('columnNameLabel')
        label.setStyleSheet("border: 4px solid #717171; font-weight: bold; color: black; font-size: 16px;")
        label.setFixedHeight(60)
        return label

    def create_result_label(self, name):
        """
        Method that creates one label used for company details
        :param name: name of the label
        :return: the label
        """
        label = QLabel(str(name))
        label.setObjectName('resultLabel')
        label.setStyleSheet("border: 2px solid #8C8C8C; color: #3B3A3A;")
        return label

    def display_top_esg_companies(self, not_first=True):
        """
        Method that displays the top list
        :param not_first: boolean
        """
        if self.top_N == self.previous_top_N and not_first:
            return

        if not_first:
            self.sound_action.play()

        self.logger.info(f"Displaying top {self.top_N} ESG companies.")

        try:
            # Initialize the Scroll area:
            top_companies = self.main_window.top_companies_df.head(self.top_N)

            self.column_names_hbox = self.display_column_names()
            new_scrollable_widget = QWidget()
            new_scrollable_widget.setStyleSheet("border: 2px solid white;")
            self.top_companies_vbox = QVBoxLayout()
            self.top_companies_vbox.addLayout(self.column_names_hbox)
            new_scrollable_widget.setLayout(self.top_companies_vbox)
            self.scrollable_area.setWidget(new_scrollable_widget)

            # Display the requested number of top companies:
            for i in range(self.top_N):
                results_hbox = QHBoxLayout()
                index_label = self.create_result_label(str(top_companies["index"].iloc[i]))
                index_label.setFixedSize(60, 40)
                index_label.setObjectName("columnNameESG")
                ticker_label = self.create_result_label(top_companies["ticker"].iloc[i])
                ticker_label.setFixedSize(60, 40)
                ticker_label.setObjectName("columnNameESG")
                stock_name_label = self.create_result_label(top_companies["stock_name"].iloc[i])
                stock_name_label.setFixedSize(200, 40)
                stock_name_label.setObjectName("columnNameESG")
                industry_label = self.create_result_label(top_companies["industry"].iloc[i])
                industry_label.setFixedSize(350, 40)
                industry_label.setObjectName("columnNameESG")

                results_hbox.setSpacing(3)
                results_hbox.addWidget(index_label)
                results_hbox.addWidget(ticker_label)
                results_hbox.addWidget(stock_name_label)
                results_hbox.addWidget(industry_label)

                self.top_companies_vbox.addLayout(results_hbox)

            source_label = QLabel("The rating is taken from "
                                                     "https://www.investors.com/news/esg-stocks-list-of-100-best-esg-companies/")
            source_label.setObjectName("sourceLabel")
            source_label.setTextInteractionFlags(source_label.textInteractionFlags() | Qt.TextSelectableByMouse)
            source_label.setFixedWidth(490)
            self.top_companies_vbox.addWidget(source_label, alignment=Qt.AlignCenter)

            self.top_companies_vbox.addStretch(1)
        except:
            self.main_window.show_error_window("Error occurred when loading the Top ESG companies.",
                                               "Please check your Internet connection.")

    def top_N_changed(self):
        """
        Method called when the user changes the desired number of companies:
        """
        top_N_entered = self.top_N_combo.currentText()
        if top_N_entered != "":
            self.previous_top_N = self.top_N
            self.top_N = int(top_N_entered)
            self.display_top_esg_companies()

    def on_finished(self):
        """
        Method called when closing the pop-up
        """
        self.sound_cancel.play()
        QTimer.singleShot(5000, self.close)

    def keyPressEvent(self, event):
        """
        Method called when a keyboard key is pressed
        :param event: the event
        """
        if event.key() not in (Qt.Key_Enter, Qt.Key_Return):
            super().keyPressEvent(event)


class CustomComboBoxESG(QComboBox):
    """
    Customized Combo Box with allowed values of 1 to 100
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)

        validator = CustomValidator(1, 100, self)
        self.lineEdit().setValidator(validator)


class CustomValidator(QValidator):
    """
    Customized validator with allowed alues of 5 to 100
    """

    def __init__(self, minimum, maximum, parent=None):
        super().__init__(parent)
        self.min = minimum
        self.max = maximum

    def validate(self, inp, pos):
        """
        Validate user input
        :param inp: input
        :param pos: position
        :return: validation results
        """
        if inp:
            try:
                value = int(inp)
                if self.min <= value <= self.max:
                    return QValidator.Acceptable, inp, pos
                else:
                    return QValidator.Invalid, inp, pos
            except ValueError:
                return QValidator.Invalid, inp, pos
        else:
            return QValidator.Intermediate, inp, pos

    def fixup(self, inp):
        """
        Method that caps values
        :param inp: user input
        :return: capped value
        """
        try:
            value = int(inp)
            if value > self.max:
                return str(self.max)
            if value < self.min:
                return str(5)
        except ValueError:
            return str(self.min)


class ErrorPopUp(QDialog):
    """
    Pop-up displayed for error messages
    """

    close_pop_up = pyqtSignal()

    def __init__(self, text_1, text_2, text_3=None):
        """
        Constructor method
        :param text_1: text for line 1
        :param text_2: text for line 2
        :param text_3: text for line 3
        """
        super().__init__()
        self.setWindowTitle("ERROR")

        # Read the style file:
        path = self.get_file_path("style.css")
        with open(path, "r") as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)

        self.setStyleSheet("QDialog { background-color: #874747; color: black; } QWidget { border: 2px solid #332020; }"
                           " QLabel { color: black; font-size: 16px; border-color: transparent; font-weight: bold; text-align: center; }"
                           " QPushButton { background-color: black; color: #FFF; font-weight: bold; outline: none; text-align: center; } "
                           "QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #AF40FF, stop:1 #00DDEB); } "
                           "QPushButton:pressed { background-color: black; }")

        # set up sounds:

        sound_directory = os.path.join(os.path.dirname(__file__), 'data/sounds')
        cancel_sound_path = os.path.join(sound_directory, 'cancel_button.wav')
        self.sound_cancel = QSoundEffect()
        self.sound_cancel.setSource(QUrl.fromLocalFile(cancel_sound_path))

        # Build the page:

        layout = QVBoxLayout()

        error_vbox = QVBoxLayout()
        error_vbox.setSpacing(20)
        error_widget = QWidget()
        error_widget.setObjectName("errorVBox")
        error_widget.setFixedSize(500, 120)
        error_widget.setLayout(error_vbox)

        error_label_1 = QLabel(text_1)
        error_label_1.setObjectName("errorLabel")
        error_label_2 = QLabel(text_2)
        error_label_2.setObjectName("errorLabel")
        error_vbox.addWidget(error_label_1, alignment=Qt.AlignCenter)
        error_vbox.addWidget(error_label_2, alignment=Qt.AlignCenter)

        if text_3 is not None:
            error_label_3 = QLabel(text_3)
            error_label_3.setObjectName("errorLabel")
            error_widget.setFixedSize(500, 180)
            error_vbox.addWidget(error_label_3, alignment=Qt.AlignCenter)

        cancel_button = QPushButton("Ok")
        cancel_button.setFixedSize(70, 40)
        cancel_button.clicked.connect(self.sound_cancel.play)
        cancel_button.clicked.connect(self.process_cancel_decision)

        layout.addWidget(error_widget)
        layout.addWidget(cancel_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def process_cancel_decision(self):
        """
        Method for closing the pop-up
        :return:
        """
        self.sound_cancel.play()
        QTimer.singleShot(100, self.close)

    def get_file_path(self, filename):
        """
        File path retrieval
        :param filename: name of file
        :return: file path
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(script_dir, 'data', filename))
        return path
