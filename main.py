import sys
from PyQt5.QtWidgets import QApplication
from MainWindow import MainWindow
import charset_normalizer.md__mypyc


# Initialize the application:
app = QApplication(sys.argv)

# Initialize the main window:
dpi = app.primaryScreen().logicalDotsPerInch()
main_window = MainWindow(dpi)

# Launch the application:
main_window.show()
sys.exit(app.exec_())

