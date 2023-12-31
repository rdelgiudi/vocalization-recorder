# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'vocWindowView.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1045, 982)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(910, 620, 88, 34))
        self.startButton.setObjectName("startButton")
        self.viewLabel = QtWidgets.QLabel(self.centralwidget)
        self.viewLabel.setGeometry(QtCore.QRect(40, 20, 960, 540))
        self.viewLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.viewLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.viewLabel.setText("")
        self.viewLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.viewLabel.setObjectName("viewLabel")
        self.switchSourceButton = QtWidgets.QPushButton(self.centralwidget)
        self.switchSourceButton.setGeometry(QtCore.QRect(800, 620, 88, 34))
        self.switchSourceButton.setObjectName("switchSourceButton")
        self.fpsLabel = QtWidgets.QLabel(self.centralwidget)
        self.fpsLabel.setGeometry(QtCore.QRect(10, 620, 41, 18))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.fpsLabel.setFont(font)
        self.fpsLabel.setObjectName("fpsLabel")
        self.fpsValLabel = QtWidgets.QLabel(self.centralwidget)
        self.fpsValLabel.setGeometry(QtCore.QRect(50, 620, 58, 18))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.fpsValLabel.setFont(font)
        self.fpsValLabel.setObjectName("fpsValLabel")
        self.timeLabel = QtWidgets.QLabel(self.centralwidget)
        self.timeLabel.setGeometry(QtCore.QRect(10, 640, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.timeLabel.setFont(font)
        self.timeLabel.setObjectName("timeLabel")
        self.timeValLabel = QtWidgets.QLabel(self.centralwidget)
        self.timeValLabel.setGeometry(QtCore.QRect(50, 640, 111, 18))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.timeValLabel.setFont(font)
        self.timeValLabel.setObjectName("timeValLabel")
        self.resolutionLabel = QtWidgets.QLabel(self.centralwidget)
        self.resolutionLabel.setGeometry(QtCore.QRect(630, 620, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.resolutionLabel.setFont(font)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.resolutionBox = QtWidgets.QComboBox(self.centralwidget)
        self.resolutionBox.setGeometry(QtCore.QRect(710, 620, 71, 31))
        self.resolutionBox.setObjectName("resolutionBox")
        self.resolutionBox.addItem("")
        self.resolutionBox.addItem("")
        self.resolutionBox.addItem("")
        self.resolutionLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.resolutionLabel_2.setGeometry(QtCore.QRect(430, 640, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.resolutionLabel_2.setFont(font)
        self.resolutionLabel_2.setObjectName("resolutionLabel_2")
        self.disparityShiftBox = QtWidgets.QSpinBox(self.centralwidget)
        self.disparityShiftBox.setGeometry(QtCore.QRect(520, 640, 71, 31))
        self.disparityShiftBox.setMaximum(200)
        self.disparityShiftBox.setObjectName("disparityShiftBox")
        self.errorLabel = QtWidgets.QLabel(self.centralwidget)
        self.errorLabel.setEnabled(True)
        self.errorLabel.setGeometry(QtCore.QRect(130, 630, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.errorLabel.setFont(font)
        self.errorLabel.setTextFormat(QtCore.Qt.AutoText)
        self.errorLabel.setObjectName("errorLabel")
        self.resolutionLabel_3 = QtWidgets.QLabel(self.centralwidget)
        self.resolutionLabel_3.setGeometry(QtCore.QRect(370, 600, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.resolutionLabel_3.setFont(font)
        self.resolutionLabel_3.setObjectName("resolutionLabel_3")
        self.histogramBox = QtWidgets.QComboBox(self.centralwidget)
        self.histogramBox.setGeometry(QtCore.QRect(520, 600, 101, 32))
        self.histogramBox.setObjectName("histogramBox")
        self.histogramBox.addItem("")
        self.histogramBox.addItem("")
        self.histogramBox.addItem("")
        self.audioComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.audioComboBox.setGeometry(QtCore.QRect(150, 680, 371, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.audioComboBox.setFont(font)
        self.audioComboBox.setObjectName("audioComboBox")
        self.audioLabel = QtWidgets.QLabel(self.centralwidget)
        self.audioLabel.setGeometry(QtCore.QRect(4, 680, 131, 21))
        self.audioLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.audioLabel.setObjectName("audioLabel")
        self.audioLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.audioLabel_2.setGeometry(QtCore.QRect(4, 720, 131, 21))
        self.audioLabel_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.audioLabel_2.setObjectName("audioLabel_2")
        self.audioComboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.audioComboBox_2.setGeometry(QtCore.QRect(150, 720, 371, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.audioComboBox_2.setFont(font)
        self.audioComboBox_2.setObjectName("audioComboBox_2")
        self.freqLabel = QtWidgets.QLabel(self.centralwidget)
        self.freqLabel.setGeometry(QtCore.QRect(550, 680, 131, 21))
        self.freqLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.freqLabel.setObjectName("freqLabel")
        self.freqBox = QtWidgets.QComboBox(self.centralwidget)
        self.freqBox.setGeometry(QtCore.QRect(690, 680, 111, 24))
        self.freqBox.setObjectName("freqBox")
        self.freqBox.addItem("")
        self.freqBox.addItem("")
        self.freqBox.addItem("")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 760, 1011, 211))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.plotLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.plotLayout.setContentsMargins(0, 0, 0, 0)
        self.plotLayout.setObjectName("plotLayout")
        self.disableSecondBox = QtWidgets.QCheckBox(self.centralwidget)
        self.disableSecondBox.setGeometry(QtCore.QRect(540, 720, 161, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.disableSecondBox.setFont(font)
        self.disableSecondBox.setObjectName("disableSecondBox")
        self.disableVideoBox = QtWidgets.QCheckBox(self.centralwidget)
        self.disableVideoBox.setGeometry(QtCore.QRect(10, 580, 161, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.disableVideoBox.setFont(font)
        self.disableVideoBox.setObjectName("disableVideoBox")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RealSense Recorder"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.switchSourceButton.setText(_translate("MainWindow", "Color"))
        self.fpsLabel.setText(_translate("MainWindow", "FPS:"))
        self.fpsValLabel.setText(_translate("MainWindow", "0"))
        self.timeLabel.setText(_translate("MainWindow", "Time:"))
        self.timeValLabel.setText(_translate("MainWindow", "0:0:0"))
        self.resolutionLabel.setText(_translate("MainWindow", "Resolution:"))
        self.resolutionBox.setItemText(0, _translate("MainWindow", "480p"))
        self.resolutionBox.setItemText(1, _translate("MainWindow", "480p (L)"))
        self.resolutionBox.setItemText(2, _translate("MainWindow", "720p"))
        self.resolutionLabel_2.setText(_translate("MainWindow", "Disparity Shift:"))
        self.errorLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#ff0000;\">ERROR: No camera device connected!</span></p></body></html>"))
        self.resolutionLabel_3.setText(_translate("MainWindow", "Histogram equalization:"))
        self.histogramBox.setItemText(0, _translate("MainWindow", "None"))
        self.histogramBox.setItemText(1, _translate("MainWindow", "Realsense"))
        self.histogramBox.setItemText(2, _translate("MainWindow", "OpenCV"))
        self.audioLabel.setText(_translate("MainWindow", "Audio device 1:"))
        self.audioLabel_2.setText(_translate("MainWindow", "Audio device 2:"))
        self.freqLabel.setText(_translate("MainWindow", "Frequency:"))
        self.freqBox.setItemText(0, _translate("MainWindow", "192 kHz"))
        self.freqBox.setItemText(1, _translate("MainWindow", "181 kHz"))
        self.freqBox.setItemText(2, _translate("MainWindow", "44.1 kHz"))
        self.disableSecondBox.setText(_translate("MainWindow", "Disable "))
        self.disableVideoBox.setText(_translate("MainWindow", "Disable Video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
