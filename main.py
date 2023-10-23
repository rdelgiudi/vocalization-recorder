# Script written for Python 3.10.10
# dependencies listed in file requirements.txt

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import sip
import vocWindowControl
import os


if __name__ == "__main__":

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    form = vocWindowControl.MainDialog(app)
    form.show()
    app.exec_()
