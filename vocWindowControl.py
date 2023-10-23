import os
import time

import matplotlib
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
import typing, datetime

import vocWindowView, vocRecordLogic

from scipy import signal
import numpy as np

#import pyqtgraph
import pyaudio

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.animation
matplotlib.use('Qt5Agg')

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
        self.disableSecondBox.clicked.connect(self.disableSecondAudioDevice)
        self.disableVideoBox.clicked.connect(self.disableVideo)

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

        # pyqtgraph.setConfigOptions(imageAxisOrder='row-major')
        # self.p1 = self.spectogram.addPlot()
        # self.img = pyqtgraph.ImageItem()
        # self.p1.addItem(self.img)
        #
        # self.hist = pyqtgraph.HistogramLUTItem()
        # self.hist.setImageItem(self.img)
        # self.spectogram.addItem(self.hist)
        #
        # self.p1.setLabel('bottom', "Time", units='s')
        # self.p1.setLabel('left', "Frequency", units='Hz')
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.plotLayout.addWidget(self.canvas)
        self.currentData = None
        self.ax = self.figure.add_subplot()
        self.figure.supxlabel("Time [s]")
        self.ax.set_ylabel("Volume")
        #self.ax.set_ylabel("Frequency [Hz]")
        self.line = None

        def animate(i):
            if self.currentData is not None:
                lock = threading.Lock()

                lock.acquire()
                audio_data = b''.join(self.currentData)
                audio_data = np.fromstring(audio_data, np.int16)
                lock.release()
                self.line.set_ydata(audio_data)
                #self.ax.draw_artist(self.mesh)
                #self.canvas.update()
                #self.canvas.flush_events()
                #self.canvas.draw()

        self.anim = matplotlib.animation.FuncAnimation(fig=self.figure, func=animate, interval=20, blit=False,
                                                       repeat=False, cache_frame_data=False)
        self.anim.pause()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.isRecording = False
        self.anim = None
        self.worker.deleteLater()
        os._exit(1)

        return super().closeEvent(a0)

    def disableSecondAudioDevice(self):
        checked = self.disableSecondBox.isChecked()

        self.audioComboBox_2.setDisabled(checked)

    def disableVideo(self):

        disableVideoChecked = self.disableVideoBox.isChecked()
        self.resolutionBox.setDisabled(disableVideoChecked)
        self.switchSourceButton.setDisabled(disableVideoChecked)
        self.histogramBox.setDisabled(disableVideoChecked)
        self.disparityShiftBox.setDisabled(disableVideoChecked)


    def startClicked(self):
        if not self.isRecording:
            self.anim.resume()
            self.isRecording = True
            self.startButton.setText("Stop")
            self.disableSecondBox.setDisabled(True)
            self.freqBox.setDisabled(True)
            self.disparityShiftBox.setDisabled(True)
            self.resolutionBox.setDisabled(True)

            if self.thread.isRunning():
                self.thread.wait()

            self.thread.start()

            # recordlogic.recording(self)
        else:
            self.anim.pause()
            self.isRecording = False
            self.errorLabel.setHidden(True)
            self.startButton.setText("Start")
            self.disableSecondBox.setDisabled(False)
            self.freqBox.setDisabled(False)
            self.disparityShiftBox.setDisabled(False)
            self.resolutionBox.setDisabled(False)
            self.thread.exit()

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
                self.dim = (640, 480)
            case 2:
                self.dim = (1280, 720)

    def disparityShiftBoxChanged(self):
        self.disparityShift = self.disparityShiftBox.value()

    def updateUi(self, depth_image_8U, color_image, sampled_audio, fps, totalseconds):

        if not self.disableVideoBox.isChecked():
            qImg = None
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

        if sampled_audio is not None:
            fs = 44100

            match self.freqBox.currentIndex():
                case 0:
                    fs = 192000
                case 1:
                    fs = 181000
                case 2:
                    fs = 44100

            audio_list = []
            lock = threading.Lock()

            if self.line is None and sampled_audio.qsize() > 100:
                for i in range(30):
                    if not sampled_audio.empty():
                        audio_list.append(sampled_audio.get())
                lock.acquire()
                self.currentData = audio_list
                lock.release()

            elif self.line is not None:
                for i in range(30):
                    if not sampled_audio.empty():
                        lock.acquire()
                        self.currentData.pop(0)
                        self.currentData.append(sampled_audio.get())
                        lock.release()

            else:
                return

            #audio_data = np.fromstring(sampled_audio, np.int16)
            audio_data = b''.join(self.currentData)
            audio_data = np.fromstring(audio_data, np.int16)

            #f, t, Sxx = signal.spectrogram(audio_data, fs)

            #frobenius_norm = np.linalg.norm(Sxx, 'nuc')
            #Sxx_float = Sxx.astype(np.float32)

            #Sxx_float = Sxx_float / frobenius_norm

            if self.line is None:
                #self.mesh = self.ax.pcolormesh(t, f, Sxx, shading="gouraud")
                start = 0
                end = (1 / fs) * len(audio_data)
                x = np.linspace(start, end, num=len(audio_data))
                self.line, = self.ax.plot(x, audio_data)
                self.ax.set_ylim([-32768, 32768])
                #self.colorbar = self.figure.colorbar(self.mesh, ax=self.ax)

            #self.currentData = [t, f, Sxx]

            # self.figure.clear()
            # self.ax = self.figure.add_subplot(111)
            # #self.ax.plot(audio_data)
            # #self.ax.set_xlabel("Sample")
            # self.ax.set_ylabel("Frequency [Hz]")
            # self.ax.pcolormesh(t, f, Sxx)
            # self.canvas.draw()