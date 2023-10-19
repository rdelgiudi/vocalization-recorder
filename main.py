import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import sip
import vocWindowControl
import os


if __name__ == "__main__":
    #ci_build_and_not_headless = False
    #try:
    #    from cv2.version import ci_build, headless
    #    ci_and_not_headless = ci_build and not headless
    #except:
    #    pass
    #if sys.platform.startswith("linux") and ci_and_not_headless:
    #    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    #if sys.platform.startswith("linux") and ci_and_not_headless:
    #    os.environ.pop("QT_QPA_FONTDIR")
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    form = vocWindowControl.MainDialog(app)
    form.show()
    app.exec_()
