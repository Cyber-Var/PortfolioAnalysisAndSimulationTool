from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QLabel, QRadioButton, QLineEdit, QComboBox, QSpacerItem, \
    QSizePolicy, QPushButton
from PyQt5.QtCore import Qt, QEvent, QSize, pyqtSignal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from UI.Page import Page


class SingleStockPage(QWidget, Page):

    back_to_portfolio_page = pyqtSignal()
    algorithm_names = ["Linear Regression", "Random Forest", "Bayesian", "Monte Carlo", "LSTM", "ARIMA"]

    def __init__(self, main_window, controller, dpi):
        super().__init__()

        self.main_window = main_window
        self.controller = controller
        self.dpi = dpi

        self.ticker = None
        self.stock_name = None
        self.one_share_price = None
        self.num_shares = None
        self.investment = None
        self.hold_duration = None

        self.left_vbox = None
        self.right_vbox = None

        self.overall_price_label = None
        # self.num_shares_combo = None

        self.hold_duration_1d = None
        self.hold_duration_1w = None
        self.hold_duration_1m = None

        self.graph_canvas = None

        self.algorithms_combo = None

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Portfolio Analysis")

        self.setStyleSheet(self.load_stylesheet())

        self.setLayout(self.layout)

    def set_parameters(self, ticker, stock_name, one_share_price, num_shares, hold_duration):
        self.ticker = ticker
        self.stock_name = stock_name
        self.one_share_price = one_share_price
        self.num_shares = num_shares
        self.investment = round(one_share_price * num_shares, 2)
        self.hold_duration = hold_duration
        self.build_page()

    def clear_layout(self, layout):
        if layout is None:
            print("None!!!")
        else:
            while layout.count():
                print("a")
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is None:
                    self.clear_layout(item.layout())
                else:
                    widget.deleteLater()

    def build_page(self):
        self.logger.info('Building the Single Stock Page')
        self.clear_layout(self.layout)

        title_label = self.get_title_label(f'{self.ticker} ({self.stock_name})')
        title_label.setFixedSize(1300, 50)
        self.layout.addWidget(title_label)

        main_hbox = QHBoxLayout()
        self.layout.addLayout(main_hbox)

        left_widget = QWidget()
        left_widget.setObjectName("singleStockVBox")
        left_widget.setFixedSize(500, 800)
        self.left_vbox = QVBoxLayout(left_widget)
        main_hbox.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setObjectName("singleStockVBox")
        right_widget.setFixedSize(800, 800)
        self.right_vbox = QVBoxLayout(right_widget)
        main_hbox.addWidget(right_widget)

        self.draw_info_and_manipulation_box()
        self.draw_algorithm_results_box()
        self.draw_graphs_box()
        self.draw_risk_metrics_box()

        back_button = QPushButton("Back")
        back_button.setObjectName("addStockButton")
        back_button.clicked.connect(self.back_to_portfolio_page.emit)
        self.layout.addWidget(back_button)

        self.layout.addStretch()

    def draw_info_and_manipulation_box(self):
        info_and_manipulation_widget = QWidget()
        info_and_manipulation_widget.setObjectName("singleStockVBox")
        info_and_manipulation_widget.setFixedSize(475, 300)
        info_and_manipulation_vbox = QVBoxLayout(info_and_manipulation_widget)
        self.left_vbox.addWidget(info_and_manipulation_widget)

        ticker_label = QLabel(f"Ticker: {self.ticker}")
        ticker_label.setObjectName("infoLabelSingleStock")
        ticker_label.setAlignment(Qt.AlignCenter)
        info_and_manipulation_vbox.addWidget(ticker_label)

        stock_name_label = QLabel(f"Stock Name: {self.stock_name}")
        stock_name_label.setObjectName("infoLabelSingleStock")
        stock_name_label.setAlignment(Qt.AlignCenter)
        info_and_manipulation_vbox.addWidget(stock_name_label)

        one_share_price_label = QLabel(f"Price of 1 share = ${self.one_share_price}")
        one_share_price_label.setObjectName("infoLabelSingleStock")
        one_share_price_label.setAlignment(Qt.AlignCenter)
        info_and_manipulation_vbox.addWidget(one_share_price_label)

        num_shares_hbox = QHBoxLayout()
        info_and_manipulation_vbox.addLayout(num_shares_hbox)

        num_shares_label = QLabel(f"Number of shares:")
        num_shares_label.setObjectName("infoLabelSingleStock")
        num_shares_label.setAlignment(Qt.AlignCenter)

        spacer_left = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_right = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)

        # self.num_shares_combo = CustomComboBox()
        # self.num_shares_combo.setFixedWidth(50)
        # self.num_shares_combo.setEditable(True)
        # self.num_shares_combo.addItem('1')
        # self.num_shares_combo.addItem('2')
        # self.num_shares_combo.addItem('3')
        # self.num_shares_combo.addItem('4')
        # self.num_shares_combo.addItem('5')
        # self.num_shares_combo.setCurrentIndex(self.num_shares - 1)
        # self.num_shares_combo.lineEdit().setPlaceholderText(f'{self.num_shares}')
        # self.num_shares_combo.activated.connect(self.num_shares_changed)
        # self.num_shares_combo.lineEdit().textChanged.connect(self.num_shares_changed)

        num_shares_result_label = QLabel(f"{self.num_shares}")
        num_shares_result_label.setObjectName("infoLabelSingleStock")
        num_shares_result_label.setAlignment(Qt.AlignCenter)
        info_and_manipulation_vbox.addWidget(num_shares_result_label)

        num_shares_hbox.addItem(spacer_left)
        num_shares_hbox.addWidget(num_shares_label)
        num_shares_hbox.addWidget(num_shares_result_label)
        num_shares_hbox.addItem(spacer_right)

        self.overall_price_label = QLabel(f"Overall price: ${self.investment}")
        self.overall_price_label.setObjectName("infoLabelSingleStock")
        self.overall_price_label.setAlignment(Qt.AlignCenter)
        info_and_manipulation_vbox.addWidget(self.overall_price_label)

        hold_duration_label = QLabel(f"Hold duration:")
        hold_duration_label.setObjectName("infoLabelSingleStock")
        hold_duration_label.setAlignment(Qt.AlignCenter)
        info_and_manipulation_vbox.addWidget(hold_duration_label)

        self.hold_duration_1d = QRadioButton("1 day")
        self.hold_duration_1d.setObjectName('inputLabel')

        self.hold_duration_1w = QRadioButton("1 week")
        self.hold_duration_1w.setObjectName('inputLabel')

        self.hold_duration_1m = QRadioButton("1 month")
        self.hold_duration_1m.setObjectName('inputLabel')

        if self.hold_duration == "1d":
            self.hold_duration_1d.setChecked(True)
        elif self.hold_duration == "1w":
            self.hold_duration_1w.setChecked(True)
        else:
            self.hold_duration_1m.setChecked(True)

        self.hold_duration_1d.toggled.connect(self.hold_duration_button_toggled)
        info_and_manipulation_vbox.addWidget(self.hold_duration_1d)

        self.hold_duration_1w.toggled.connect(self.hold_duration_button_toggled)
        info_and_manipulation_vbox.addWidget(self.hold_duration_1w)

        self.hold_duration_1m.toggled.connect(self.hold_duration_button_toggled)
        info_and_manipulation_vbox.addWidget(self.hold_duration_1m)

    def draw_algorithm_results_box(self):
        algorithm_results_widget = QWidget()
        algorithm_results_widget.setObjectName("singleStockVBox")
        algorithm_results_widget.setFixedSize(475, 465)
        algorithm_results_vbox = QVBoxLayout(algorithm_results_widget)
        self.left_vbox.addWidget(algorithm_results_widget)

        self.algorithms_combo = CustomComboBox()
        self.algorithms_combo.addItem("Please select an algorithm")
        for algorithm_name in self.algorithm_names:
            self.algorithms_combo.addItem(algorithm_name)
        # self.algorithms_combo.addItem('Linear Regression')
        # self.algorithms_combo.addItem('Random Forest')
        # self.algorithms_combo.addItem('Bayesian')
        # self.algorithms_combo.addItem('Monte Carlo')
        # self.algorithms_combo.addItem('LSTM')
        # self.algorithms_combo.addItem('ARIMA')
        self.algorithms_combo.activated.connect(self.algorithm_changed)
        self.algorithms_combo.setCurrentIndex(0)
        self.algorithms_combo.setFixedSize(200, 50)
        algorithm_results_vbox.addWidget(self.algorithms_combo)

        self.algorithm_results = [self.controller.linear_regression_results, self.controller.random_forest_results,
                                  self.controller.bayesian_results, self.controller.monte_carlo_results,
                                  self.controller.lstm_results, self.controller.arima_results]
        self.algorithm_predicted_prices = [self.controller.linear_regression_predicted_prices, self.controller.random_forest_predicted_prices,
                                  self.controller.bayesian_predicted_prices, self.controller.monte_carlo_predicted_prices,
                                  self.controller.lstm_predicted_prices, self.controller.arima_predicted_prices]

        predicted_price_hbox = QHBoxLayout()
        algorithm_results_vbox.addLayout(predicted_price_hbox)

        self.predicted_price_label = QLabel("Predicted Price:")
        self.predicted_price_label.setObjectName("algorithmResultSingleStock")
        self.predicted_price_label.hide()
        predicted_price_hbox.addWidget(self.predicted_price_label)

        self.predicted_price_result_label = QLabel()
        self.predicted_price_result_label.setObjectName("algorithmResultSingleStock")
        self.predicted_price_result_label.hide()
        predicted_price_hbox.addWidget(self.predicted_price_result_label)

        profit_loss_hbox = QHBoxLayout()
        algorithm_results_vbox.addLayout(profit_loss_hbox)

        self.profit_loss_label = QLabel()
        self.profit_loss_label.setObjectName("algorithmResultSingleStock")
        self.profit_loss_label.hide()
        profit_loss_hbox.addWidget(self.profit_loss_label)

        self.profit_loss_result_label = QLabel()
        self.profit_loss_result_label.setObjectName("algorithmResultSingleStock")
        self.profit_loss_result_label.hide()
        profit_loss_hbox.addWidget(self.profit_loss_result_label)

    def draw_graphs_box(self):
        graphs_widget = QWidget()
        graphs_widget.setObjectName("singleStockVBox")
        graphs_widget.setFixedSize(775, 645)
        graphs_vbox = QVBoxLayout(graphs_widget)
        self.right_vbox.addWidget(graphs_widget)

        graphs_choice_hbox = QHBoxLayout()
        graphs_vbox.addLayout(graphs_choice_hbox)

        self.history_graph_radio = QRadioButton("Historical Price")
        self.history_graph_radio.setObjectName('inputLabel')
        self.history_graph_radio.setFixedSize(200, 50)
        self.history_graph_radio.setChecked(True)
        self.history_graph_radio.toggled.connect(self.graphs_choice_button_toggled)
        graphs_choice_hbox.addWidget(self.history_graph_radio)

        self.moving_average_graph_radio = QRadioButton("Moving Average")
        self.moving_average_graph_radio.setObjectName('inputLabel')
        self.moving_average_graph_radio.setFixedSize(200, 50)
        self.moving_average_graph_radio.toggled.connect(self.graphs_choice_button_toggled)
        graphs_choice_hbox.addWidget(self.moving_average_graph_radio)

        self.arima_graph_radio = QRadioButton("ARIMA")
        self.arima_graph_radio.setObjectName('inputLabel')
        self.arima_graph_radio.setFixedSize(200, 50)
        self.arima_graph_radio.toggled.connect(self.graphs_choice_button_toggled)
        graphs_choice_hbox.addWidget(self.arima_graph_radio)

        self.monte_carlo_graph_radio = QRadioButton("Monte Carlo Simulation")
        self.monte_carlo_graph_radio.setObjectName('inputLabel')
        self.monte_carlo_graph_radio.setFixedSize(200, 50)
        self.monte_carlo_graph_radio.toggled.connect(self.graphs_choice_button_toggled)
        graphs_choice_hbox.addWidget(self.monte_carlo_graph_radio)

        self.graph_figure = Figure(figsize=(700, 550), dpi=self.dpi)
        self.graph_figure = self.controller.plot_historical_price_data(self.ticker, self.hold_duration, self.graph_figure)
        self.graph_canvas = FigureCanvas(self.graph_figure)
        graphs_vbox.addWidget(self.graph_canvas)

    def draw_risk_metrics_box(self):
        risk_metrics_widget = QWidget()
        risk_metrics_widget.setObjectName("singleStockVBox")
        risk_metrics_widget.setFixedSize(775, 125)
        risk_metrics_hbox = QHBoxLayout(risk_metrics_widget)
        self.right_vbox.addWidget(risk_metrics_widget)

        vol, sharpe, VaR = self.controller.get_risk_metrics(self.ticker)
        volatility, volatility_category = vol
        sharpe_ratio, sharpe_ratio_categpry = sharpe

        volatility_label = QLabel(f"Volatility\n{volatility:.2f} {volatility_category}")
        volatility_label.setObjectName("riskMetricLabel")
        volatility_label.setFixedSize(200, 70)
        risk_metrics_hbox.addWidget(volatility_label)

        sharpe_ratio_label = QLabel(f"Sharpe Ratio\n{sharpe_ratio:.2f} {sharpe_ratio_categpry}")
        sharpe_ratio_label.setObjectName("riskMetricLabel")
        sharpe_ratio_label.setFixedSize(200, 70)
        risk_metrics_hbox.addWidget(sharpe_ratio_label)

        VaR_label = QLabel(f"Value at Risk\n{VaR:.2f}")
        VaR_label.setObjectName("riskMetricLabel")
        VaR_label.setFixedSize(200, 70)
        risk_metrics_hbox.addWidget(VaR_label)

    # TODO: processing for this change
    def hold_duration_button_toggled(self, checked):
        if checked:
            self.logger.info('Handling the change in hold duration.')
            if self.hold_duration_1d.isChecked():
                self.hold_duration = "1d"
                self.update_graph()
                self.algorithm_changed()
            elif self.hold_duration_1w.isChecked():
                self.hold_duration = "1w"
                self.update_graph()
                self.algorithm_changed()
            elif self.hold_duration_1m.isChecked():
                self.hold_duration = "1m"
                self.update_graph()
                self.algorithm_changed()

    def graphs_choice_button_toggled(self, checked):
        if checked:
            self.logger.info('Handling the change in choice of graph.')
            if (self.history_graph_radio.isChecked() or self.moving_average_graph_radio.isChecked() or
                    self.arima_graph_radio.isChecked() or self.monte_carlo_graph_radio.isChecked()):
                self.update_graph()

    def update_graph(self):
        self.graph_figure.clear()
        if self.history_graph_radio.isChecked():
            self.graph_figure = self.controller.plot_historical_price_data(self.ticker, self.hold_duration, self.graph_figure)
        elif self.moving_average_graph_radio.isChecked():
            self.graph_figure = self.controller.plotMovingAverage(self.ticker, self.hold_duration, self.graph_figure)
        elif self.arima_graph_radio.isChecked():
            self.algorithms_combo.setCurrentIndex(6)
            self.algorithm_changed()
            # if self.ticker not in self.controller.arima_results[self.hold_duration].keys():
            #     self.controller.run_arima(self.ticker, self.hold_duration)
            self.graph_figure = self.controller.plotARIMA(self.ticker, self.hold_duration, self.graph_figure)
        else:
            self.algorithms_combo.setCurrentIndex(4)
            self.algorithm_changed()
            # if self.ticker not in self.controller.monte_carlo_results[self.hold_duration].keys():
            #     self.controller.run_monte_carlo(self.ticker, self.hold_duration)
            self.graph_figure = self.controller.plot_monte_carlo(self.ticker, self.hold_duration, self.graph_figure)
        self.graph_canvas.draw()

    # def num_shares_changed(self):
    #     num_shares_entered = self.num_shares_combo.currentText()
    #     if num_shares_entered == "":
    #         self.overall_price_label.setText(f"Overall price: -")
    #         # TODO: when this is -, do not allow calculations or anything - make pop up error message if user tries
    #     else:
    #         self.num_shares = int(num_shares_entered)
    #         self.investment = round(self.one_share_price * self.num_shares, 2)
    #         self.overall_price_label.setText(f"Overall price: ${self.investment}")
    #
    #         self.controller.update_investment_amount(self.ticker, self.investment)

    def algorithm_changed(self):
        algorithm_entered = self.algorithms_combo.currentText()

        if algorithm_entered == "Please select an algorithm":
            self.predicted_price_label.hide()
            self.predicted_price_result_label.hide()
            self.profit_loss_label.hide()
            self.profit_loss_result_label.hide()
        else:
            algorithm_index = self.algorithm_names.index(algorithm_entered)

            if self.ticker not in self.algorithm_results[algorithm_index][self.hold_duration].keys():
                self.controller.run_algorithm(self.ticker, algorithm_index, self.hold_duration)

            if algorithm_index == 2:
                self.predicted_price_result_label.setText(
                    f"${self.algorithm_predicted_prices[algorithm_index][self.hold_duration][self.ticker]:.2f}"
                    f" +/- {self.controller.bayesian_confidences[self.hold_duration][self.ticker][0]:.2f}")
            elif algorithm_index == 5:
                self.arima_graph_radio.setChecked(True)
                self.predicted_price_result_label.setText(
                    f"${self.algorithm_predicted_prices[algorithm_index][self.hold_duration][self.ticker]:.2f}"
                    f" +/- {self.controller.arima_confidences[self.hold_duration][self.ticker][0]:.2f}")
            else:
                self.predicted_price_result_label.setText(
                    f"${self.algorithm_predicted_prices[algorithm_index][self.hold_duration][self.ticker]:.2f}")

            if algorithm_index == 3:
                self.monte_carlo_graph_radio.setChecked(True)
                self.profit_loss_result_label.setText(self.algorithm_results[algorithm_index][self.hold_duration][self.ticker])
            else:
                profit_loss = self.algorithm_results[algorithm_index][self.hold_duration][self.ticker]
                if profit_loss >= 0:
                    self.profit_loss_label.setText("Profit amount:")
                    self.profit_loss_result_label.setText(f"+${profit_loss:.2f}")
                else:
                    self.profit_loss_label.setText("Loss amount:")
                    self.profit_loss_result_label.setText(f"-${abs(profit_loss):.2f}")

                if algorithm_index == 2:
                    self.profit_loss_result_label.setText(
                        f"{self.algorithm_results[algorithm_index][self.hold_duration][self.ticker]:.2f}"
                        f" +/- {abs(self.controller.bayesian_confidences[self.hold_duration][self.ticker][1]):.2f}")
                elif algorithm_index == 5:
                    self.profit_loss_result_label.setText(
                        f"${self.algorithm_results[algorithm_index][self.hold_duration][self.ticker]:.2f}"
                        f" +/- {abs(self.controller.arima_confidences[self.hold_duration][self.ticker][1]):.2f}")
                self.profit_loss_label.show()

            self.predicted_price_label.show()
            self.predicted_price_result_label.show()
            self.profit_loss_result_label.show()

    def open_menu_page(self):
        # TODO: create logic
        self.logger.info(f'Opening the Portfolio Page')


class CustomComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Return or key == Qt.Key_Enter:
                text = self.currentText()
                if text.isdigit() and int(text) > 0:
                    self.lineEdit().setText(text)
                else:
                    self.lineEdit().clear()
            elif not (key == Qt.Key_Backspace or key == Qt.Key_Delete or key == Qt.Key_Tab or
                      key == Qt.Key_Left or key == Qt.Key_Right or key == Qt.Key_Home or key == Qt.Key_End):
                if not (48 <= key <= 57):
                    return True
        return super().eventFilter(obj, event)
