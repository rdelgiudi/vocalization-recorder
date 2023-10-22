import json
import multiprocessing
import time

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import datetime
from PyQt5 import QtGui
import cv2
import os
from multiprocessing import Process, Queue
#from billiard import Process, Queue
#import billiard
import threading
import pyaudio
import wave
from scipy.io.wavfile import write

def processWQueue(colorwqueue, depthwqueue, dims, dateandtime):

    #dateandtime = datetime.datetime.today().isoformat("_", "seconds")
    #dateandtime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    output_dir = os.path.join(os.getcwd(), "videos")

    color_path = os.path.join(output_dir, "{}_rgb.avi".format(dateandtime))
    depth_path = os.path.join(output_dir, "{}_depth.avi".format(dateandtime))

    print(f"[INFO] Saving video of dimensions {dims[0]} x {dims[1]}")
    print(f"[INFO] Saving color stream to: {color_path}")
    print(f"[INFO] Saving depth stream to: {depth_path}")
    colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*"XVID"), 30, dims, 1)
    depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*"DIVX"), 30, dims, 0)

    while True:
        if colorwqueue.empty() and depthwqueue.empty():
            continue
        colordata = colorwqueue.get()
        depthdata = depthwqueue.get()
        if type(colordata) == str and type(depthdata) == str:
            break

        if type(depthdata) != str:
            depthwriter.write(depthdata)
        if type(colordata) != str:
            colorwriter.write(colordata)

def processAWQueue(worker, first_data, second_data, first_wave_file, second_wave_file, info_queue, data_queue):
    FORMAT = pyaudio.paInt16  # We use 16bit format per sample
    CHANNELS = 2
    RATE = 44100
    match worker.window.freqBox.currentIndex():
        case 0:
            RATE = 192000
        case 1:
            RATE = 181000
        case 2:
            RATE = 44100

    print("Sampling rate: {} Hz".format(RATE))

    CHUNK = 1024*4
    DEVICE_FIRST = 0
    DEVICE_SECOND = 0

    audio = pyaudio.PyAudio()

    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    temp_iter = 0

    selected_index_first = worker.window.audioComboBox.currentIndex()
    print("[DEBUG] Selected index first: {}".format(selected_index_first))

    selected_index_second = worker.window.audioComboBox_2.currentIndex()
    print("[DEBUG] Selected index second: {}".format(selected_index_second))

    second_audio_disabled = worker.window.disableSecondBox.isChecked()

    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            if (temp_iter == selected_index_first):
                DEVICE_FIRST = i
                print("[DEBUG] Device first selected")
            if (temp_iter == selected_index_second):
                DEVICE_SECOND = i
                print("[DEBUG] Device second selected")

            temp_iter += 1

    print("Selected 1st Input Device id ", DEVICE_FIRST, " - ",
          audio.get_device_info_by_host_api_device_index(0, DEVICE_FIRST).get('name'))

    print("Selected 2nd Input Device id ", DEVICE_SECOND, " - ",
          audio.get_device_info_by_host_api_device_index(0, DEVICE_SECOND).get('name'))

    stream_first = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              output=False,
                              #frames_per_buffer=CHUNK,
                              input_device_index=DEVICE_FIRST
                              )
    if not second_audio_disabled:
        stream_second = audio.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=False,
                                  #frames_per_buffer=CHUNK,
                                  input_device_index=DEVICE_SECOND
                                  )

    first_wave_file.setnchannels(CHANNELS)
    first_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    first_wave_file.setframerate(RATE)

    if not second_audio_disabled:
        second_wave_file.setnchannels(CHANNELS)
        second_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        second_wave_file.setframerate(RATE)

    try:
        while True:
            if not info_queue.empty():

                data = info_queue.get()

                if data == "DONE":
                    break

            audio_first = stream_first.read(CHUNK)
            data_queue.put(audio_first)
            if not second_audio_disabled:
                audio_second = stream_second.read(CHUNK)

            first_data.append(audio_first)
            if not second_audio_disabled:
                second_data.append(audio_second)

    finally:
        stream_first.stop_stream()
        stream_first.close()
        if not second_audio_disabled:
            stream_second.stop_stream()
            stream_second.close()
        audio.terminate()

def progress_callback(progress):
    print(f'\rProgress  {progress}% ... ', end ="\r")

def config_video(worker, dateandtime):

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)

    try:
        pipeline_profile = config.resolve(pipeline_wrapper)
    except:
        print("ERROR: No device connected")
        worker.window.errorLabel.setHidden(False)
        return

    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The program requires Depth camera with Color sensor")
        exit(0)

    # Set Disparity Shift
    # adv_mode = rs.rs400_advanced_mode(device)
    # depth_table_control_group = adv_mode.get_depth_table()
    # depth_table_control_group.disparityShift = worker.window.disparityShift

    # Set colorizer settings
    #colorizer = rs.colorizer()
    # if worker.window.histogramBox.currentIndex() == 1:
    #     colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
    # else:
    #     colorizer.set_option(rs.option.histogram_equalization_enabled, 0)

    dims = worker.window.dim

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, dims[0], dims[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, dims[0], dims[1], rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()

    # Depth unit, set to equivalent of 100 in RealSense Viewer
    #if depth_sensor.supports(rs.option.depth_units):
    #    depth_sensor.set_option(rs.option.depth_units, 0.0001)

    #depth_scale = depth_sensor.get_depth_scale()

    colorwqueue = Queue()
    depthwqueue = Queue()

    writethread = Process(target=processWQueue, args=(colorwqueue, depthwqueue, dims, dateandtime))
    writethread.start()

    return pipeline, align, colorwqueue, depthwqueue, writethread

def recording(worker):

    pipeline = None
    align = None
    colorwqueue = None
    depthwqueue = None
    writethread = None

    disableVideo = worker.window.disableVideoBox.isChecked()

    dateandtime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    if not disableVideo:
        pipeline, align, colorwqueue, depthwqueue, writethread = config_video(worker, dateandtime)

    second_audio_disabled = worker.window.disableSecondBox.isChecked()

    #dateandtime = datetime.datetime.today().isoformat("-", "seconds")
    output_dir = os.path.join(os.getcwd(), "videos")
    first_path = os.path.join(output_dir, "{}_first.wav".format(dateandtime))
    second_path = os.path.join(output_dir, "{}_second.wav".format(dateandtime))

    first_wave_file = wave.open(first_path, 'wb')

    print("Saving first wave to file: {}".format(first_path))

    if not second_audio_disabled:
        second_wave_file = wave.open(second_path, 'wb')
        print("Saving second wave to file: {}".format(second_path))
    else:
        second_wave_file = None

    # create output folder if it doesn't exist
    folder_name = "videos"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(f"Folder {folder_name} created successfully!")
    else:
        print(f"Folder {folder_name} already exists.")

    framenum = 0
    fps = 0
    starttime = datetime.datetime.now()
    fpstime = starttime

    capturetimes = []
    aligntimes = []
    numpytimes = []
    #writetimes = []
    #updateuitimes = []

    info_queue = Queue()
    data_queue = Queue()
    first_data = []
    second_data = []

    color_image = None
    depth_image_8U = None

    audio_write_thread = threading.Thread(target=processAWQueue, args=(worker, first_data, second_data,
                                                                       first_wave_file, second_wave_file, info_queue, data_queue))

    audio_write_thread.start()

    try:
        while worker.window.isRecording:

            #if not worker.window.isRecording:
            #    break

            if not disableVideo:
                if (datetime.datetime.now()-fpstime).total_seconds() > 5:
                    framenum = 0
                    fpstime = datetime.datetime.now()

                # Wait for a coherent pair of frames: depth and color
                capturestart = datetime.datetime.now()
                frames = pipeline.wait_for_frames()

                capturened = datetime.datetime.now()
                capturetimes.append((capturened - capturestart).total_seconds())

                alignstart = datetime.datetime.now()
                aligned_frames = align.process(frames)
                alignend = datetime.datetime.now()
                aligntimes.append((alignend - alignstart).total_seconds())

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Filters (comment in and out for best effect)
                #depth_frame = rs.hole_filling_filter().process(depth_frame)
                #depth_frame = colorizer.colorize(depth_frame)

                numpystart = datetime.datetime.now()
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_image_8U = cv2.convertScaleAbs(depth_image, alpha=0.03)

                # Built in opencv histogram equalization
                if worker.window.histogramBox.currentIndex() == 2:
                    depth_image_8U = cv2.equalizeHist(depth_image_8U)

                numpyend = datetime.datetime.now()
                numpytimes.append((numpyend - numpystart).total_seconds())

                colorwqueue.put(color_image)
                depthwqueue.put(depth_image_8U)

                framenum += 1
                currenttime = datetime.datetime.now()
                fpsseconds = (currenttime - fpstime).total_seconds()
                fps = framenum / fpsseconds
            else:
                time.sleep(0.01)
                currenttime = datetime.datetime.now()

            totalseconds = (currenttime - starttime).total_seconds()

            if not data_queue.empty():
                worker.progress.emit(depth_image_8U, color_image, data_queue, fps, totalseconds)
            else:
                worker.progress.emit(depth_image_8U, color_image, None, fps, totalseconds)

    finally:
        print("Waiting for writing process to finish...")

        # Stop streaming
        if not disableVideo:
            pipeline.stop()
            colorwqueue.put("DONE")
            depthwqueue.put("DONE")

        info_queue.put("DONE")
        if not disableVideo:
            writethread.join()
            colorwqueue.close()
            depthwqueue.close()

        audio_write_thread.join()

        first_wave_file.writeframes(b''.join(first_data))
        first_wave_file.close()

        if not second_audio_disabled:
           second_wave_file.writeframes(b''.join(second_data))
           second_wave_file.close()

        data_queue.close()
        info_queue.close()

        print("Done!")

        if not disableVideo:
            print("FPS: {:.2f}".format(fps))
            print("Operation Times:")
            captureavg = sum(capturetimes) / len(capturetimes)
            alignavg = sum(aligntimes) / len(aligntimes)
            numpyavg = sum(numpytimes) / len(numpytimes)

            print("Capture {} ms".format(captureavg * 1000))
            print("Align: {} ms".format(alignavg * 1000))
            print("Numpy: {} ms".format(numpyavg * 1000))



