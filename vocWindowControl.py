import os
import typing
import datetime

import numpy as np
import pyaudio

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation
matplotlib.use('Qt5Agg')

import cv2

import vocWindowView
import vocRecordLogic

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

        self.startButton.clicked.connect(self.start_clicked)
        self.switchSourceButton.clicked.connect(self.switch_source_clicked)
        self.resolutionBox.currentIndexChanged.connect(self.resolution_box_changed)
        self.disparityShiftBox.valueChanged.connect(self.disparity_shift_box_changed)
        self.disableSecondBox.clicked.connect(self.disable_second_audio_device)
        self.disableVideoBox.clicked.connect(self.disable_video)
        self.freqBox.currentIndexChanged.connect(self.change_fs)

        self.isRecording = False
        self.showDepth = False
        self.disparityShift = 0
        self.dim = (848, 480)
        self.fs = 192000

        self.thread = QThread()
        self.worker = Worker(window=self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_ui)

        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
                self.audioComboBox.addItem(p.get_device_info_by_host_api_device_index(0, i).get('name'))
                self.audioComboBox_2.addItem(p.get_device_info_by_host_api_device_index(0, i).get('name'))

        p.terminate()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.plotLayout.addWidget(self.canvas)
        self.current_data = None
        self.ax = self.figure.add_subplot()
        self.figure.supxlabel("Time [s]")
        self.ax.set_ylabel("Intensity")
        self.line = None

        def animate(i):
            if self.current_data is not None:
                audio_data = b''.join(self.current_data)
                audio_data = np.fromstring(audio_data, np.int16)
                self.line.set_ydata(audio_data)

        self.anim = matplotlib.animation.FuncAnimation(fig=self.figure, func=animate, interval=20, blit=False,
                                                       repeat=False, cache_frame_data=False)
        self.anim.pause()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.isRecording = False
        self.anim = None
        self.worker.deleteLater()
        #os._exit(1)

        return super().closeEvent(a0)
    def disable_second_audio_device(self):
        checked = self.disableSecondBox.isChecked()

        self.audioComboBox_2.setDisabled(checked)

    def change_fs(self):
        match self.freqBox.currentIndex():
            case 0:
                self.fs = 192000
            case 1:
                self.fs = 181000
            case 2:
                self.fs = 44100

    def disable_video(self):
        disable_video_checked = self.disableVideoBox.isChecked()
        self.resolutionBox.setDisabled(disable_video_checked)
        self.switchSourceButton.setDisabled(disable_video_checked)
        self.histogramBox.setDisabled(disable_video_checked)
        self.disparityShiftBox.setDisabled(disable_video_checked)

    def start_clicked(self):
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

    def switch_source_clicked(self):
        if not self.showDepth:
            self.showDepth = True
            self.switchSourceButton.setText("Depth")
        else:
            self.showDepth = False
            self.switchSourceButton.setText("Color")

    def resolution_box_changed(self):
        match self.resolutionBox.currentIndex():
            case 0:
                self.dim = (848, 480)
            case 1:
                self.dim = (640, 480)
            case 2:
                self.dim = (1280, 720)

    def disparity_shift_box_changed(self):
        self.disparityShift = self.disparityShiftBox.value()

    def update_ui(self, depth_image_8U, color_image, sampled_audio, fps, total_seconds):

        if not self.disableVideoBox.isChecked():
            qImg = None
            if self.showDepth:
                qt_img = depth_image_8U
                height, width = qt_img.shape
                bytes_per_line = width
                qImg = QImage(qt_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                qt_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                height, width, channel = qt_img.shape
                bytes_per_line = 3 * width
                qImg = QImage(qt_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap01 = QPixmap.fromImage(qImg)
            self.viewLabel.setPixmap(pixmap01.scaled(960, 540, Qt.KeepAspectRatio))
            self.fpsValLabel.setText("{:.2f}".format(fps))

        self.timeValLabel.setText(str(datetime.timedelta(seconds=total_seconds)))

        if sampled_audio is not None:
            audio_list = []

            if self.line is None and sampled_audio.qsize() > 10:
                for i in range(9):
                    if not sampled_audio.empty():
                        audio_list.append(sampled_audio.get())

                self.current_data = audio_list

            elif self.line is not None:
                for i in range(9):
                    if not sampled_audio.empty():
                        self.current_data.pop(0)
                        self.current_data.append(sampled_audio.get())

            else:
                return

            audio_data = b''.join(self.current_data)
            audio_data = np.fromstring(audio_data, np.int16)

            if self.line is None:
                start = 0
                end = (1 / self.fs) * len(audio_data)
                x = np.linspace(start, end, num=len(audio_data))
                self.line, = self.ax.plot(x, audio_data)
                self.ax.set_ylim([-32768, 32768])
