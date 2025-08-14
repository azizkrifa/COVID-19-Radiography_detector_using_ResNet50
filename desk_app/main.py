import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PySide6.QtWidgets import QApplication, QMainWindow
from mainwindow import MainWindowUI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = MainWindowUI()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
