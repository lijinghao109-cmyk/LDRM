"""
main.py - Entry point for LocalDoc Research Manager (LDRM).

Usage:
    python main.py

Requirements (install via pip):
    pip install -r requirements.txt
"""

import sys
import os

# Ensure the project root is on the Python path so all modules resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

import db
from ui_main import MainWindow


def main():
    # 1. Bootstrap the database (creates data/app.db + table if missing)
    db.init_db()

    # 2. Launch the Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("LocalDoc 研究管理器")
    app.setOrganizationName("LDRM")
    app.setStyle("Fusion")  # consistent look across platforms

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
