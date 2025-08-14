from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QSizePolicy, QHBoxLayout
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt
from model import CovidModel

class MainWindowUI:
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("COVID-19 Radiography Detector")
        MainWindow.resize(550, 550)
        self.central_widget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.central_widget)

        # Set dark blue background for the main window
        self.central_widget.setStyleSheet("""
            background: #0d1a3a;
        """)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(28)
        self.layout.setContentsMargins(40, 40, 40, 40)

        # Center image_label horizontally
        image_hbox = QHBoxLayout()
        image_hbox.addStretch(1)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 3px solid #1565c0;
            background: #1e2a47;
            min-height: 300px;
            min-width: 300px;
            max-height: 400px;
            max-width: 400px;
            font-size: 18px;
            color: #bbb;
            border-radius: 14px;
        """)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        image_hbox.addWidget(self.image_label)
        image_hbox.addStretch(1)
        self.layout.addLayout(image_hbox)

         # Connections
        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedWidth(250)
        self.load_button.setStyleSheet("""
             QPushButton {
                padding: 12px;
                font-size: 17px;
                background: #1565c0;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                margin-top: 12px;
            }
            QPushButton:hover {
                background: #0d47a1;
                color: #bbdefb;
            }
        """)

        btn_hbox = QHBoxLayout()
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(self.load_button)
        btn_hbox.addStretch(1)
        self.layout.addLayout(btn_hbox)

        self.load_button.clicked.connect(self.load_image)

        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            margin-top: 0px;
            border-radius: 8px;
            padding: 0px;
            font-size: 22px;
        """)
        self.layout.addWidget(self.result_label)

        # Model loading
        self.model = CovidModel("./output/best_model.h5")  # Ensure model.h5 is in desk_app folder

        self.image_path = None
        self.full_image = False  # Track toggle state

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.central_widget, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.full_image = False
            self.update_image_display()
            self.predict_image()

    def update_image_display(self):
        if not self.image_path:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No image loaded")
            return
        size = 500 if self.full_image else 290
        pixmap = QPixmap(self.image_path).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")

    def predict_image(self):
        if not self.image_path:
            self.result_label.setText("Please load an image first.")
            return
        label, confidence = self.model.predict(self.image_path)
        self.result_label.setText(
            f"<span style='color:#fff; font-size:24px'>{label}</span><br>"
            f"<span style='color:#c8e6c9; font-size:20px'>Confidence: {confidence:.2%}</span>"
        )
