import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Thread
import typing, datetime

import vocWindowView, vocRecordLogic

from scipy import signal
import numpy as np

import pyqtgraph
import pyaudio
import matplotlib.pyplot as plt

import cv2


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(object, object, object, float, float)

    def __init__(self, window, parent: typing.Optional['QObject'] = ...) -> None:
        super().__init__()
        self.window = window

    def run(self):
        vocRecordLogic.recording(self)
        self.finished.emit()


class MainDialog(QMainWindow, vocWindowView.Ui_MainWindow):

    def __init__(self, app, parent=None):
        super(MainDialog, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(QSize(1045, 982))
        self.app = app

        self.errorLabel.setHidden(True)
        self.histogramBox.setCurrentIndex(2)

        self.startButton.clicked.connect(self.startClicked)
        self.switchSourceButton.clicked.connect(self.switchSourceClicked)
        self.resolutionBox.currentIndexChanged.connect(self.resolutionBoxChanged)
        self.disparityShiftBox.valueChanged.connect(self.disparityShiftBoxChanged)

        self.isRecording = False
        self.showDepth = False
        self.disparityShift = 0
        self.dim = (848, 480)

        self.thread = QThread()
        self.worker = Worker(window=self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.updateUi)

        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
                self.audioComboBox.addItem(p.get_device_info_by_host_api_device_index(0, i).get('name'))
                self.audioComboBox_2.addItem(p.get_device_info_by_host_api_device_index(0, i).get('name'))

        p.terminate()

        pyqtgraph.setConfigOptions(imageAxisOrder='row-major')
        self.p1 = self.spectogram.addPlot()
        self.img = pyqtgraph.ImageItem()
        self.p1.addItem(self.img)

        self.hist = pyqtgraph.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.spectogram.addItem(self.hist)

        self.p1.setLabel('bottom', "Time", units='s')
        self.p1.setLabel('left', "Frequency", units='Hz')


    def closeEvent(self, a0: QCloseEvent) -> None:

        self.isRecording = False

        return super().closeEvent(a0)

    def startClicked(self):
        if not self.isRecording:
            self.isRecording = True
            self.startButton.setText("Stop")

            if self.thread.isRunning():
                self.thread.wait()

            self.thread.start()

            # recordlogic.recording(self)
        else:
            self.isRecording = False
            self.errorLabel.setHidden(True)
            self.startButton.setText("Start")

    def switchSourceClicked(self):
        if not self.showDepth:
            self.showDepth = True
            self.switchSourceButton.setText("Depth")
        else:
            self.showDepth = False
            self.switchSourceButton.setText("Color")

    def resolutionBoxChanged(self):
        match self.resolutionBox.currentIndex():
            case 0:
                self.dim = (848, 480)
            case 1:
                self.dim = (1280, 720)

    def disparityShiftBoxChanged(self):
        self.disparityShift = self.disparityShiftBox.value()

    def updateUi(self, depth_image_8U, color_image, sampled_audio, fps, totalseconds):

        if self.showDepth:
            qtimg = depth_image_8U
            height, width = qtimg.shape
            bytesPerLine = width
            qImg = QImage(qtimg.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        else:
            qtimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            height, width, channel = qtimg.shape
            bytesPerLine = 3 * width
            qImg = QImage(qtimg.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap01 = QPixmap.fromImage(qImg)

        self.viewLabel.setPixmap(pixmap01.scaled(960, 540, Qt.KeepAspectRatio))

        self.fpsValLabel.setText("{:.2f}".format(fps))
        self.timeValLabel.setText(str(datetime.timedelta(seconds=totalseconds)))

        # if sampled_audio is not None:
        #     fs = 44100
        #
        #     match self.freqBox.currentIndex():
        #         case 0:
        #             fs = 181000
        #         case 1:
        #             fs = 44100
        #
        #     audio_data = np.fromstring(sampled_audio, np.int16)
        #
        #     #f, t, Sxx = signal.spectrogram(audio_data, fs)
        #
        #     data = np.fft.rfft(audio_data)
        #
        #     self.hist.setLevels(np.min(data), np.max(data))
        #
        #     self.hist.gradient.restoreState(
        #         {'mode': 'rgb',
        #          'ticks': [(0.5, (0, 182, 188, 255)),
        #                    (1.0, (246, 111, 0, 255)),
        #                    (0.0, (75, 0, 113, 255))]})
        #
        #     self.img.setImage(data)
        #     #self.img.scale(t[-1]/np.size(Sxx, axis=1), f[-1]/np.size(Sxx, axis=0))
        #     #self.p1.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])