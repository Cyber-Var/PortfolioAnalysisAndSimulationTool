from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QLabel, QRadioButton, QLineEdit, QComboBox, QSpacerItem, \
    QSizePolicy, QPushButton
from PyQt5.QtCore import Qt, QEvent, QSize

from UI.Page import Page


class SingleStockPage(QWidget, Page):

    def __init__(self, main_window, controller, ticker, stock_name, one_share_price, num_shares, hold_duration):
        super().__init__()

        self.main_window = main_window
        self.controller = controller
        self.ticker = ticker
        self.stock_name = stock_name
        self.one_share_price = one_share_price
        self.num_shares = num_shares
        self.hold_duration = hold_duration

        self.left_vbox = None
        self.right_vbox = None

        self.overall_price_label = None
        self.num_shares_combo = None

        self.hold_duration_1d = None
        self.hold_duration_1w = None
        self.hold_duration_1m = None

        self.investment = round(one_share_price * num_shares, 2)

        self.setStyleSheet(self.load_stylesheet())

        self.layout = QVBoxLayout()
        self.init_page("Portfolio Analysis")

        self.build_page()

        self.setStyleSheet(self.load_stylesheet())

        self.setLayout(self.layout)

    def build_page(self):
        self.logger.info('Building the Single Stock Page')

        title_label = self.get_title_label(f'{self.ticker} ({self.stock_name})')
        title_label.setFixedSize(1300, 50)
        self.layout.addWidget(title_label)

        main_hbox = QHBoxLayout()
        self.layout.addLayout(main_hbox)

        # line_widget = LineDrawingWidget()
        # line_widget.setFixedSize(1300, 850)
        # self.layout.addWidget(line_widget)

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
        back_button.clicked.connect(self.open_menu_page)
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
        # num_shares_hbox.addWidget(num_shares_label)

        spacer_left = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_right = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.num_shares_combo = CustomComboBox()
        self.num_shares_combo.setFixedWidth(50)
        self.num_shares_combo.setEditable(True)
        self.num_shares_combo.lineEdit().setPlaceholderText(f'{self.num_shares}')
        self.num_shares_combo.addItem('1')
        self.num_shares_combo.addItem('2')
        self.num_shares_combo.addItem('3')
        self.num_shares_combo.addItem('4')
        self.num_shares_combo.addItem('5')
        self.num_shares_combo.setCurrentIndex(self.num_shares)
        self.num_shares_combo.activated.connect(self.num_shares_changed)
        self.num_shares_combo.lineEdit().textChanged.connect(self.num_shares_changed)
        # num_shares_hbox.addWidget(self.num_shares_combo)

        num_shares_hbox.addItem(spacer_left)
        num_shares_hbox.addWidget(num_shares_label)
        num_shares_hbox.addWidget(self.num_shares_combo)
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
        self.hold_duration_1d.toggled.connect(self.hold_duration_button_toggled)
        info_and_manipulation_vbox.addWidget(self.hold_duration_1d)

        self.hold_duration_1w = QRadioButton("1 week")
        self.hold_duration_1w.setObjectName('inputLabel')
        self.hold_duration_1w.toggled.connect(self.hold_duration_button_toggled)
        info_and_manipulation_vbox.addWidget(self.hold_duration_1w)

        self.hold_duration_1m = QRadioButton("1 month")
        self.hold_duration_1m.setObjectName('inputLabel')
        self.hold_duration_1m.toggled.connect(self.hold_duration_button_toggled)
        info_and_manipulation_vbox.addWidget(self.hold_duration_1m)

        if self.hold_duration == "1d":
            self.hold_duration_1d.setChecked(True)
        elif self.hold_duration == "1w":
            self.hold_duration_1w.setChecked(True)
        else:
            self.hold_duration_1m.setChecked(True)

    def draw_algorithm_results_box(self):
        algorithm_results_widget = QWidget()
        algorithm_results_widget.setObjectName("singleStockVBox")
        algorithm_results_widget.setFixedSize(475, 465)
        algorithm_results_vbox = QVBoxLayout(algorithm_results_widget)
        self.left_vbox.addWidget(algorithm_results_widget)

    def draw_graphs_box(self):
        graphs_widget = QWidget()
        graphs_widget.setObjectName("singleStockVBox")
        graphs_widget.setFixedSize(775, 645)
        graphs_vbox = QVBoxLayout(graphs_widget)
        self.right_vbox.addWidget(graphs_widget)

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
                print("d")
                self.hold_duration = "1d"
            elif self.hold_duration_1w.isChecked():
                print("w")
                self.hold_duration = "1w"
            elif self.hold_duration_1m.isChecked():
                print("m")
                self.hold_duration = "1m"

    def num_shares_changed(self):
        num_shares_entered = self.num_shares_combo.currentText()
        if num_shares_entered == "":
            self.overall_price_label.setText(f"Overall price: -")
            # TODO: when this is -, do not allow calculations or anything - make pop up error message if user tries
        else:
            self.num_shares = int(num_shares_entered)
            self.investment = round(self.one_share_price * self.num_shares, 2)
            self.overall_price_label.setText(f"Overall price: ${self.investment}")

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

# class LineDrawingWidget(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.main_hbox = QHBoxLayout(self)
#         self.setLayout(self.main_hbox)
#         self.setMinimumSize(QSize(1300, 850))
#
#     def paintEvent(self, event):
#         super().paintEvent(event)
#         print("Drawable size:", self.width(), self.height())
#         painter = QPainter(self)
#         painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
#         painter.drawLine(500, 0, 500, self.height())


# class LineDrawer(QWidget):
#     coordinates = None
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.coordinates = (0, 0, 100, 100)
#
#     def setCoordinates(self, x0, y0, x1, y1):
#         self.coordinates = (x0, y0, x1, y1)
#         self.update()
#
#     def paintEvent(self, event):
#         if self.coordinates:
#             painter = QPainter(self)
#             painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
#             painter.drawLine(*self.coordinates)


