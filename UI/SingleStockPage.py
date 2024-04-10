from PyQt5.QtGui import QPainter, QPen, QFont, QColor, QBrush
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QLabel, QRadioButton, QLineEdit, QComboBox, QSpacerItem, \
    QSizePolicy, QPushButton, QTableWidget, QTableWidgetItem, QAbstractItemView, QFrame
from PyQt5.QtCore import Qt, QEvent, QSize, pyqtSignal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from UI.Page import Page


class SingleStockPage(QWidget, Page):

    back_to_portfolio_page = pyqtSignal()
    algorithm_names = ["Linear Regression", "Random Forest", "Bayesian", "Monte Carlo Simulation", "LSTM", "ARIMA"]
    color_green = QColor('#00FF00')
    color_red = QColor('#FF0000')

    def __init__(self, main_window, controller, dpi):
        super().__init__()

        self.graph_figure = None
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

    def set_parameters(self, ticker, stock_name, one_share_price, num_shares, hold_duration, is_long):
        self.ticker = ticker
        self.stock_name = stock_name
        self.one_share_price = one_share_price
        self.num_shares = num_shares
        self.investment = round(one_share_price * num_shares, 2)
        self.hold_duration = hold_duration
        self.is_long = is_long
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

        title_label = self.get_title_label(f'{self.ticker} ({self.stock_name})', "titleLabelSingleStock")
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
        back_button.setFixedSize(90, 40)
        back_button.clicked.connect(self.back_to_portfolio_page.emit)
        self.layout.addWidget(back_button)

        self.layout.addStretch()

    def draw_info_and_manipulation_box(self):
        info_and_manipulation_widget = QWidget()
        info_and_manipulation_widget.setObjectName("singleStockVBox")
        info_and_manipulation_widget.setFixedSize(475, 340)
        info_and_manipulation_vbox = QVBoxLayout(info_and_manipulation_widget)
        self.left_vbox.addWidget(info_and_manipulation_widget)

        info_table = QTableWidget()
        info_table.setRowCount(6)
        info_table.setColumnCount(2)
        info_table.setFixedSize(300, 222)

        info_table.setColumnWidth(0, 150)
        info_table.setColumnWidth(1, 200)
        for i in range(6):
            info_table.setRowHeight(i, 37)

        bold_font = QFont()
        bold_font.setBold(True)

        large_font = QFont()
        large_font.setPointSize(20)
        large_font.setBold(True)

        larger_font = QFont()
        larger_font.setPointSize(25)
        larger_font.setBold(True)

        ticker_label_1 = QTableWidgetItem("Ticker")
        ticker_label_1.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(0, 0, ticker_label_1)
        ticker_label_1.setFont(bold_font)

        ticker_label_2 = QTableWidgetItem(self.ticker)
        ticker_label_2.setTextAlignment(Qt.AlignCenter)
        color_1 = QColor('#EF3054')
        ticker_label_2.setForeground(QBrush(color_1))
        info_table.setItem(0, 1, ticker_label_2)
        ticker_label_2.setFont(larger_font)

        stock_name_label_1 = QTableWidgetItem("Stock Name")
        stock_name_label_1.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(1, 0, stock_name_label_1)
        stock_name_label_1.setFont(bold_font)

        stock_name_label_2 = QTableWidgetItem(self.stock_name)
        stock_name_label_2.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(1, 1, stock_name_label_2)
        stock_name_label_2.setFont(large_font)

        one_share_price_label_1 = QTableWidgetItem("Price of 1 share")
        one_share_price_label_1.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(2, 0, one_share_price_label_1)
        one_share_price_label_1.setFont(bold_font)

        one_share_price_label_2 = QTableWidgetItem(f"${self.one_share_price}")
        one_share_price_label_2.setTextAlignment(Qt.AlignCenter)
        one_share_price_label_2.setForeground(QBrush(color_1))
        info_table.setItem(2, 1, one_share_price_label_2)
        one_share_price_label_2.setFont(large_font)

        num_shares_label_1 = QTableWidgetItem("Number of shares")
        num_shares_label_1.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(3, 0, num_shares_label_1)
        num_shares_label_1.setFont(bold_font)

        num_shares_label_2 = QTableWidgetItem(str(self.num_shares))
        num_shares_label_2.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(3, 1, num_shares_label_2)
        num_shares_label_2.setFont(large_font)

        investment_type_label_1 = QTableWidgetItem("Investment type")
        investment_type_label_1.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(4, 0, investment_type_label_1)
        investment_type_label_1.setFont(bold_font)

        if self.is_long:
            investment_type = "Long"
        else:
            investment_type = "Short"
        investment_type_label_2 = QTableWidgetItem(investment_type)
        investment_type_label_2.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(4, 1, investment_type_label_2)
        investment_type_label_2.setFont(large_font)

        overall_price_label_1 = QTableWidgetItem("Overall price")
        overall_price_label_1.setTextAlignment(Qt.AlignCenter)
        info_table.setItem(5, 0, overall_price_label_1)
        overall_price_label_1.setFont(bold_font)

        overall_price_label_2 = QTableWidgetItem(f"${self.investment}")
        overall_price_label_2.setTextAlignment(Qt.AlignCenter)
        overall_price_label_2.setForeground(QBrush(color_1))
        info_table.setItem(5, 1, overall_price_label_2)
        overall_price_label_2.setFont(large_font)

        self.stylize_table(info_table)

        info_table.setStyleSheet("QTableWidget { background-color: black; color: white; gridline-color: white; font-size: 15px; }")
        info_and_manipulation_vbox.addWidget(info_table, alignment=Qt.AlignCenter)

        hold_duration_vbox = QVBoxLayout()

        hold_duration_label = QLabel(f"Hold duration:")
        hold_duration_label.setObjectName("inputHeaderLabel")
        hold_duration_label.setAlignment(Qt.AlignCenter)

        hold_duration_hbox = QHBoxLayout()
        hold_duration_widget = QWidget()
        hold_duration_widget.setObjectName("inputHBox")
        hold_duration_widget.setFixedSize(250, 80)
        hold_duration_widget.setLayout(hold_duration_hbox)

        self.hold_duration_1d = self.create_hold_duration_button("1 day")
        self.hold_duration_1w = self.create_hold_duration_button("1 week")
        self.hold_duration_1m = self.create_hold_duration_button("1 month")

        if self.hold_duration == "1d":
            self.hold_duration_1d.setChecked(True)
        elif self.hold_duration == "1w":
            self.hold_duration_1w.setChecked(True)
        else:
            self.hold_duration_1m.setChecked(True)

        self.hold_duration_1d.toggled.connect(self.hold_duration_button_toggled)
        self.hold_duration_1w.toggled.connect(self.hold_duration_button_toggled)
        self.hold_duration_1m.toggled.connect(self.hold_duration_button_toggled)

        hold_duration_vbox.addWidget(self.hold_duration_1d)
        hold_duration_vbox.addWidget(self.hold_duration_1w)
        hold_duration_vbox.addWidget(self.hold_duration_1m)

        hold_duration_hbox.addWidget(hold_duration_label)
        hold_duration_hbox.addLayout(hold_duration_vbox)

        info_and_manipulation_vbox.addWidget(hold_duration_widget, alignment=Qt.AlignCenter)

    def create_hold_duration_button(self, name):
        button = QRadioButton(name)
        button.setObjectName('inputLabel')
        return button

    def draw_algorithm_results_box(self):
        algorithm_results_widget = QWidget()
        algorithm_results_widget.setObjectName("singleStockVBox")
        algorithm_results_widget.setFixedSize(475, 425)
        self.algorithm_results_vbox = QVBoxLayout(algorithm_results_widget)
        self.algorithm_results_vbox.setSpacing(20)
        self.left_vbox.addWidget(algorithm_results_widget)

        self.algorithms_combo = CustomComboBox()
        self.algorithms_combo.addItem("Please select an algorithm")
        for algorithm_name in self.algorithm_names:
            self.algorithms_combo.addItem(algorithm_name)
        self.algorithms_combo.activated.connect(self.algorithm_changed)
        self.algorithms_combo.setCurrentIndex(0)
        self.algorithms_combo.setFixedSize(350, 50)
        self.algorithm_results_vbox.addWidget(self.algorithms_combo, alignment=Qt.AlignCenter)

        self.results_table = QTableWidget()
        self.results_table.setRowCount(2)
        self.results_table.setColumnCount(2)
        self.results_table.hide()

        self.large_font = QFont()
        self.large_font.setPointSize(20)
        self.large_font.setBold(True)

        predicted_price_label_1 = QTableWidgetItem("Predicted Price")
        predicted_price_label_1.setTextAlignment(Qt.AlignCenter)
        self.results_table.setItem(0, 0, predicted_price_label_1)

        predicted_price_label = QTableWidgetItem("")
        predicted_price_label.setTextAlignment(Qt.AlignCenter)
        self.results_table.setItem(0, 1, predicted_price_label)

        profit_loss_label_1 = QTableWidgetItem("")
        profit_loss_label_1.setTextAlignment(Qt.AlignCenter)
        self.results_table.setItem(1, 0, profit_loss_label_1)

        profit_loss_label = QTableWidgetItem("")
        profit_loss_label.setTextAlignment(Qt.AlignCenter)
        self.results_table.setItem(1, 1, profit_loss_label)

        self.setDefaultResultsTable()
        self.stylize_table(self.results_table)
        self.algorithm_results_vbox.addWidget(self.results_table, alignment=Qt.AlignCenter)


        self.algorithm_results = [self.controller.linear_regression_results, self.controller.random_forest_results,
                                  self.controller.bayesian_results, self.controller.monte_carlo_results,
                                  self.controller.lstm_results, self.controller.arima_results]
        self.algorithm_predicted_prices = [self.controller.linear_regression_predicted_prices, self.controller.random_forest_predicted_prices,
                                  self.controller.bayesian_predicted_prices, self.controller.monte_carlo_predicted_prices,
                                  self.controller.lstm_predicted_prices, self.controller.arima_predicted_prices]

        monte_probabilities_hbox = QHBoxLayout()

        self.fall_widget = QWidget()
        self.fall_widget.setObjectName("singleStockVBox")
        self.fall_widget.setFixedSize(200, 220)
        self.fall_probabilities_vbox = QVBoxLayout(self.fall_widget)
        self.fall_widget.setLayout(self.fall_probabilities_vbox)
        self.fall_widget.hide()

        self.growth_widget = QWidget()
        self.growth_widget.setObjectName("singleStockVBox")
        self.growth_widget.setFixedSize(220, 220)
        self.growth_probabilities_vbox = QVBoxLayout(self.growth_widget)
        self.growth_widget.setLayout(self.growth_probabilities_vbox)
        self.growth_widget.hide()

        monte_probabilities_hbox.addStretch(1)
        monte_probabilities_hbox.addWidget(self.fall_widget)
        monte_probabilities_hbox.addWidget(self.growth_widget)
        monte_probabilities_hbox.addStretch(1)

        self.smaller_font = QFont()
        self.smaller_font.setPointSize(15)
        self.smaller_font.setBold(True)

        for i in range(8):
            probability_label_1 = QLabel()
            probability_label_1.setObjectName("monteProbabilityGreen")
            probability_label_1.hide()
            self.growth_probabilities_vbox.addWidget(probability_label_1)

            probability_label_2 = QLabel()
            probability_label_2.setObjectName("monteProbabilityRed")
            probability_label_2.hide()
            self.fall_probabilities_vbox.addWidget(probability_label_2)

        self.algorithm_results_vbox.addLayout(monte_probabilities_hbox)
        self.algorithm_results_vbox.addStretch(1)

    def setDefaultResultsTable(self):
        self.results_table.setFixedSize(400, 150)
        self.results_table.setColumnWidth(0, 200)
        self.results_table.setColumnWidth(1, 200)
        self.results_table.setRowHeight(0, 75)
        self.results_table.setRowHeight(1, 75)
        self.results_table.item(0, 0).setFont(self.large_font)
        self.results_table.item(0, 1).setFont(self.large_font)
        color_2 = QColor('#125E8A')
        self.results_table.item(0, 1).setForeground(QBrush(color_2))
        self.results_table.item(1, 0).setFont(self.large_font)
        self.results_table.item(1, 1).setFont(self.large_font)

    def setMonteResultsTable(self):
        self.results_table.setFixedSize(400, 70)
        self.results_table.setColumnWidth(0, 150)
        self.results_table.setColumnWidth(1, 250)
        self.results_table.setRowHeight(0, 35)
        self.results_table.setRowHeight(1, 35)
        self.results_table.item(0, 0).setFont(self.smaller_font)
        self.results_table.item(0, 1).setFont(self.smaller_font)
        self.results_table.item(1, 0).setFont(self.smaller_font)
        self.results_table.item(1, 1).setFont(self.smaller_font)

    def stylize_table(self, table):
        table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(Qt.NoFocus)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setStyleSheet("QTableWidget { background-color: black; color: white; gridline-color: white; font-size: 15px; }")

    def draw_graphs_box(self):
        graphs_widget = QWidget()
        graphs_widget.setObjectName("singleStockVBox")
        graphs_widget.setFixedSize(775, 545)
        graphs_vbox = QVBoxLayout(graphs_widget)
        self.right_vbox.addWidget(graphs_widget)

        graphs_choice_hbox = QHBoxLayout()
        graphs_vbox.addLayout(graphs_choice_hbox)

        self.history_graph_radio = QRadioButton("Historical Price")
        self.history_graph_radio.setObjectName('inputLabel')
        self.history_graph_radio.setFixedSize(200, 40)
        self.history_graph_radio.setChecked(True)
        self.history_graph_radio.toggled.connect(self.graphs_choice_button_toggled)

        self.moving_average_graph_radio = QRadioButton("Moving Average")
        self.moving_average_graph_radio.setObjectName('inputLabel')
        self.moving_average_graph_radio.setFixedSize(200, 40)
        self.moving_average_graph_radio.toggled.connect(self.graphs_choice_button_toggled)

        self.arima_graph_radio = QRadioButton("ARIMA")
        self.arima_graph_radio.setObjectName('inputLabel')
        self.arima_graph_radio.setFixedSize(200, 40)
        self.arima_graph_radio.toggled.connect(self.graphs_choice_button_toggled)

        graphs_choice_hbox.addWidget(self.history_graph_radio)
        graphs_choice_hbox.addWidget(self.moving_average_graph_radio)
        graphs_choice_hbox.addWidget(self.arima_graph_radio)

        self.graph_figure = Figure(figsize=(700, 450), dpi=self.dpi)
        self.graph_figure = self.controller.plot_historical_price_data(self.ticker, self.hold_duration, self.graph_figure)
        self.graph_canvas = FigureCanvas(self.graph_figure)
        graphs_vbox.addWidget(self.graph_canvas)

    def draw_risk_metrics_box(self):
        risk_metrics_widget = QWidget()
        risk_metrics_widget.setObjectName("singleStockVBox")
        risk_metrics_widget.setFixedSize(775, 225)
        self.right_vbox.addWidget(risk_metrics_widget)

        risk_metrics_vbox = QVBoxLayout(risk_metrics_widget)

        risk_metrics_hbox = QHBoxLayout()

        vol, sharpe, VaR = self.controller.get_risk_metrics(self.ticker)
        volatility, volatility_category = vol
        sharpe_ratio, sharpe_ratio_categpry = sharpe

        volatility_frame, volatility_frame_layout, volatility_frame_layout_bottom = self.create_frame()
        volatility_top_label = QLabel(f"Volatility")
        volatility_top_label.setObjectName("riskMetricTopLabel")
        volatility_top_label.setFixedSize(100, 30)

        volatility_label = QLabel(f"{volatility:.2f}")
        volatility_label.setObjectName("riskMetricBottomLabel")
        volatility_label.setFixedSize(50, 30)
        volatility_label.setStyleSheet(self.process_risk_metric_style(volatility_category))
        volatility_label.setAlignment(Qt.AlignCenter)
        volatility_frame_layout_bottom.addWidget(volatility_label)

        volatility_category_label = QLabel(volatility_category)
        volatility_category_label.setObjectName("riskMetricBottomLabel")
        volatility_category_label.setFixedSize(80, 40)
        volatility_category_label.setAlignment(Qt.AlignCenter)
        volatility_category_label.setStyleSheet(self.process_category_style(volatility_category))
        volatility_frame_layout_bottom.addWidget(volatility_category_label)

        volatility_frame_layout.addWidget(volatility_top_label, alignment=Qt.AlignCenter)
        volatility_frame_layout.addLayout(volatility_frame_layout_bottom)
        volatility_frame.setFixedSize(200, 90)
        risk_metrics_hbox.addWidget(volatility_frame)

        sharpe_frame, sharpe_frame_layout, sharpe_frame_layout_bottom = self.create_frame()
        sharpe_top_label = QLabel(f"Sharpe Ratio")
        sharpe_top_label.setObjectName("riskMetricTopLabel")
        sharpe_top_label.setFixedSize(150, 30)

        sharpe_label = QLabel(f"{sharpe_ratio:.2f}")
        sharpe_label.setObjectName("riskMetricBottomLabel")
        sharpe_label.setFixedSize(50, 30)
        sharpe_label.setStyleSheet(self.process_risk_metric_style(sharpe_ratio_categpry))
        sharpe_label.setAlignment(Qt.AlignCenter)
        sharpe_frame_layout_bottom.addWidget(sharpe_label)

        sharpe_category_label = QLabel(sharpe_ratio_categpry)
        sharpe_category_label.setObjectName("riskMetricBottomLabel")
        sharpe_category_label.setFixedSize(80, 40)
        sharpe_category_label.setAlignment(Qt.AlignCenter)
        sharpe_category_label.setStyleSheet(self.process_category_style(sharpe_ratio_categpry))
        sharpe_frame_layout_bottom.addWidget(sharpe_category_label)

        sharpe_frame_layout.addWidget(sharpe_top_label, alignment=Qt.AlignCenter)
        sharpe_frame_layout.addLayout(sharpe_frame_layout_bottom)
        sharpe_frame.setFixedSize(200, 90)
        risk_metrics_hbox.addWidget(sharpe_frame)

        VaR_frame, VaR_frame_layout, VaR_frame_layout_bottom = self.create_frame()
        VaR_top_label = QLabel(f"Value at Risk")
        VaR_top_label.setObjectName("riskMetricTopLabel")
        VaR_top_label.setFixedSize(150, 30)

        VaR_label = QLabel(f"${VaR:.2f}")
        VaR_label.setObjectName("riskMetricTopLabel")
        VaR_label.setFixedSize(100, 30)
        VaR_style = self.process_risk_metric_style("Normal")
        VaR_label.setStyleSheet(f"{VaR_style} font-size: 22px;")
        VaR_label.setAlignment(Qt.AlignCenter)

        VaR_frame_layout.addWidget(VaR_top_label, alignment=Qt.AlignCenter)
        VaR_frame_layout.addWidget(VaR_label, alignment=Qt.AlignCenter)
        VaR_frame.setFixedSize(200, 90)
        risk_metrics_hbox.addWidget(VaR_frame)

        esg_hbox = QHBoxLayout()

        total_score, e_score, s_score, g_score = self.controller.get_esg_scores(self.ticker)
        esg_frame, esg_frame_layout = self.create_esg_frame()

        esg_top_label = QLabel(f"ESG Score = {total_score}")
        esg_top_label.setObjectName("riskMetricTopLabel")
        esg_top_label.setFixedSize(220, 30)

        # esg_bottom_label = QLabel(f"E-Score = {e_score}, S-Score = {s_score}, G-Score = {g_score}")
        esg_bottom_label = QLabel(f"<font color='red'>E-Score = {e_score},</font>\t<font color='green'>S-Score = {s_score},</font>\t<font color='blue'>G-Score = {g_score}</font>")
        esg_bottom_label.setObjectName("riskMetricBottomLabel")
        esg_bottom_label.setFixedSize(450, 30)
        # esg_bottom_label.setAlignment(Qt.AlignCenter)

        esg_frame_layout.addWidget(esg_top_label, alignment=Qt.AlignCenter)
        esg_frame_layout.addWidget(esg_bottom_label, alignment=Qt.AlignCenter)
        esg_frame.setFixedSize(600, 90)
        esg_hbox.addWidget(esg_frame)

        risk_metrics_vbox.addLayout(risk_metrics_hbox)
        risk_metrics_vbox.addLayout(esg_hbox)

    def create_frame(self):
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(0)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame.setObjectName("riskMetricFrame")

        frame_layout_bottom = QHBoxLayout()
        frame_layout_bottom.setSpacing(0)
        frame_layout_bottom.setContentsMargins(5, 5, 5, 5)
        return frame, frame_layout, frame_layout_bottom

    def create_esg_frame(self):
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(0)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame.setObjectName("riskMetricFrame")
        return frame, frame_layout

    def process_category_style(self, category):
        if category == "Low":
            return ("font-size: 20px; font-weight: bold; color: #8EF9F3; border: 2px solid #8EF9F3; border-radius: "
                    "5px; padding: 5px;")
        elif category == "Normal":
            return ("font-size: 20px; font-weight: bold; color: #A882DD; border: 2px solid #A882DD; border-radius: "
                    "5px; padding: 5px;")
        return ("font-size: 20px; font-weight: bold; color: #FF5733; border: 2px solid #FF5733; border-radius: 5px; "
                "padding: 5px;")

    def process_risk_metric_style(self, category):
        if category == "Low":
            return "color: #8EF9F3;"
        elif category == "Normal":
            return "color: #A882DD;"
        return "color: #FF5733;"

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
                    self.arima_graph_radio.isChecked()):
                self.update_graph()

    def update_graph(self):
        self.graph_figure.clear()
        if self.history_graph_radio.isChecked():
            self.logger.info("Displaying the History Price graph.")
            self.graph_figure = self.controller.plot_historical_price_data(self.ticker, self.hold_duration, self.graph_figure)
        elif self.moving_average_graph_radio.isChecked():
            self.logger.info("Displaying the Moving Average graph.")
            self.graph_figure = self.controller.plotMovingAverage(self.ticker, self.hold_duration, self.graph_figure)
        else:
            self.logger.info("Displaying the ARIMA graph.")
            self.algorithms_combo.setCurrentIndex(6)
            self.algorithm_changed()
            self.graph_figure = self.controller.plotARIMA(self.ticker, self.hold_duration, self.graph_figure)
        self.graph_canvas.draw()

    def algorithm_changed(self):
        algorithm_entered = self.algorithms_combo.currentText()

        if self.growth_probabilities_vbox.itemAt(0).widget().isVisible():
            for i in range(8):
                self.growth_probabilities_vbox.itemAt(i).widget().hide()
                self.fall_probabilities_vbox.itemAt(i).widget().hide()

        if algorithm_entered == "Please select an algorithm":
            self.results_table.hide()
            self.growth_widget.hide()
            self.fall_widget.hide()
        else:
            algorithm_index = self.algorithm_names.index(algorithm_entered)
            self.logger.info(f"Displaying results for algorithm with index {algorithm_index}")


            if self.ticker not in self.algorithm_results[algorithm_index][self.hold_duration].keys():
                self.controller.run_algorithm(self.ticker, algorithm_index, self.hold_duration)

            if algorithm_index == 2:
                self.results_table.item(0, 1).setText(
                    f"${self.algorithm_predicted_prices[algorithm_index][self.hold_duration][self.ticker]:.2f}"
                    f" +/- {self.controller.bayesian_confidences[self.hold_duration][self.ticker][0]:.2f}")
            elif algorithm_index == 5:
                self.arima_graph_radio.setChecked(True)
                self.results_table.item(0, 1).setText(
                    f"${self.algorithm_predicted_prices[algorithm_index][self.hold_duration][self.ticker]:.2f}"
                    f" +/- {self.controller.arima_confidences[self.hold_duration][self.ticker][0]:.2f}")
            else:
                self.results_table.item(0, 1).setText(
                    f"${self.algorithm_predicted_prices[algorithm_index][self.hold_duration][self.ticker]:.2f}")

            if algorithm_index == 3:
                self.setMonteResultsTable()
                self.growth_widget.show()
                self.fall_widget.show()

                monte_result = self.algorithm_results[algorithm_index][self.hold_duration][self.ticker]
                monte_result_splitted = monte_result.split()
                percentage = float(monte_result_splitted[0][:-1])
                last_word = monte_result_splitted[-1]

                if (last_word == "growth" and self.is_long) or (last_word == "fall" and not self.is_long):
                    self.results_table.item(1, 1).setText(monte_result + " (profit)")
                    self.results_table.item(1, 1).setForeground(QBrush(self.color_green))
                else:
                    self.results_table.item(1, 1).setText(monte_result + " (loss)")
                    self.results_table.item(1, 1).setForeground(QBrush(self.color_red))

                self.results_table.item(1, 0).setText("Prediction")

                if last_word == "growth":
                    self.growth_probabilities_vbox.itemAt(0).widget().setText(f"Chance of growth: {percentage}%")
                    self.fall_probabilities_vbox.itemAt(0).widget().setText(f"Chance of fall: {100 - percentage}%")
                else:
                    self.growth_probabilities_vbox.itemAt(0).widget().setText(f"Chance of growth: {100 - percentage}%")
                    self.fall_probabilities_vbox.itemAt(0).widget().setText(f"Chance of fall: {percentage}%")
                self.growth_probabilities_vbox.itemAt(0).widget().show()
                self.fall_probabilities_vbox.itemAt(0).widget().show()

                growth_probs, fall_probs = self.controller.get_monte_carlo_probabilities(self.ticker, self.hold_duration)
                for i in range(len(growth_probs)):
                    if i + 1 == 8:
                        break
                    self.growth_probabilities_vbox.itemAt(i + 1).widget().setText(growth_probs[i])
                    self.growth_probabilities_vbox.itemAt(i + 1).widget().show()
                for i in range(len(fall_probs)):
                    if i + 1 == 8:
                        break
                    self.fall_probabilities_vbox.itemAt(i + 1).widget().setText(fall_probs[i])
                    self.fall_probabilities_vbox.itemAt(i + 1).widget().show()
                self.growth_probabilities_vbox.addStretch(1)
                self.fall_probabilities_vbox.addStretch(1)
            else:
                self.setDefaultResultsTable()
                self.growth_widget.hide()
                self.fall_widget.hide()

                profit_loss = self.algorithm_results[algorithm_index][self.hold_duration][self.ticker]
                if profit_loss >= 0:
                    self.results_table.item(1, 0).setText("Profit amount")
                    str_result = f"+${profit_loss:.2f}"
                    self.results_table.item(1, 1).setForeground(QBrush(self.color_green))
                else:
                    self.results_table.item(1, 0).setText("Loss amount")
                    str_result = f"-${abs(profit_loss):.2f}"
                    self.results_table.item(1, 1).setForeground(QBrush(self.color_red))

                if algorithm_index == 2:
                    str_result += f" +/- {abs(self.controller.bayesian_confidences[self.hold_duration][self.ticker][1]):.2f}"
                elif algorithm_index == 5:
                    str_result += f" +/- {abs(self.controller.arima_confidences[self.hold_duration][self.ticker][1]):.2f}"

                self.results_table.item(1, 1).setText(str_result)

            self.results_table.show()

        self.algorithm_results_vbox.addStretch(1)


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
