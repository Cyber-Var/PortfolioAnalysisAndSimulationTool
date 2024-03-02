from UI.MainWindow import MainWindow


class PortfolioPage(MainWindow):
    def __init__(self):
        super().__init__()

        logger = logging.getLogger(__name__)

        self.setWindowTitle('Portfolio Analysis Page')
