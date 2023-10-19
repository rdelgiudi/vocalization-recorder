import json
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

def processWQueue(colorwqueue, depthwqueue, dims):

    dateandtime = datetime.datetime.today().isoformat("_", "seconds")

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
            RATE = 128000
        case 1:
            RATE = 44100

    print("Sampling rate: {} Hz".format(RATE))

    CHUNK = 1024
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

    stream_second = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              output=False,
                              # frames_per_buffer=CHUNK,
                              input_device_index=DEVICE_SECOND
                              )

    first_wave_file.setnchannels(CHANNELS)
    first_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    first_wave_file.setframerate(RATE)

    second_wave_file.setnchannels(CHANNELS)
    second_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    second_wave_file.setframerate(RATE)

    while True:
        if not info_queue.empty():

            data = info_queue.get()

            if data == "DONE":
                stream_first.stop_stream()
                stream_first.close()
                audio.terminate()
                break

        audio_first = stream_first.read(CHUNK)
        data_queue.put(audio_first)
        audio_second = stream_second.read(CHUNK)

        first_data.append(audio_first)
        second_data.append(audio_second)

def progress_callback(progress):
    print(f'\rProgress  {progress}% ... ', end ="\r")

def recording(worker):

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    dateandtime = datetime.datetime.today().isoformat("-", "seconds")
    output_dir = os.path.join(os.getcwd(), "videos")
    first_path = os.path.join(output_dir, "{}_first.wav".format(dateandtime))
    second_path = os.path.join(output_dir, "{}_second.wav".format(dateandtime))

    first_wave_file = wave.open(first_path, 'wb')
    second_wave_file = wave.open(second_path, 'wb')

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
    adv_mode = rs.rs400_advanced_mode(device)
    depth_table_control_group = adv_mode.get_depth_table()
    depth_table_control_group.disparityShift = worker.window.disparityShift

    # Set colorizer settings
    colorizer = rs.colorizer()
    if worker.window.histogramBox.currentIndex() == 1:
        colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
    else:
        colorizer.set_option(rs.option.histogram_equalization_enabled, 0)

    dims = worker.window.dim

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, dims[0], dims[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, dims[0], dims[1], rs.format.bgr8, 30)

    # create output folder if it doesn't exist
    folder_name = "videos"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(f"Folder {folder_name} created successfully!")
    else:
        print(f"Folder {folder_name} already exists.")

    # Start streaming
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()

    # Depth unit, set to equivalent of 100 in RealSense Viewer
    if depth_sensor.supports(rs.option.depth_units):
        depth_sensor.set_option(rs.option.depth_units, 0.0001)

    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    framenum = 0
    fps = 0
    starttime = datetime.datetime.now()
    fpstime = starttime

    capturetimes = []
    aligntimes = []
    numpytimes = []
    #writetimes = []
    #updateuitimes = []

    colorwqueue = Queue()
    depthwqueue = Queue()

    writethread = Process(target=processWQueue, args=(colorwqueue, depthwqueue, dims))
    writethread.start()

    info_queue = Queue()
    data_queue = Queue()
    first_data = []
    second_data = []

    audio_write_thread = threading.Thread(target=processAWQueue, args=(worker, first_data, second_data,
                                                                       first_wave_file, second_wave_file, info_queue, data_queue))
    audio_write_thread.start()

    try:
        while worker.window.isRecording:

            #if not worker.window.isRecording:
            #    break

            if (datetime.datetime.now()-fpstime).total_seconds() > 5:
                framenum = 0
                fpstime = datetime.datetime.now()

            # Wait for a coherent pair of frames: depth and color
            capturestart = datetime.datetime.now()
            frames = pipeline.wait_for_frames()
            #info_queue.put("OK")

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

            #writestart = datetime.datetime.now()
            #colorwriter.write(color_image)
            #depthwriter.write(depth_image_8U)
            colorwqueue.put(color_image)
            depthwqueue.put(depth_image_8U)
            #writeend = datetime.datetime.now()
            #writetimes.append((writeend - writestart).total_seconds())

            #updateuistart = datetime.datetime.now()
            framenum += 1
            currenttime = datetime.datetime.now()
            totalseconds = (currenttime - starttime).total_seconds()
            fpsseconds = (currenttime - fpstime).total_seconds()
            fps = framenum / fpsseconds

            #Thread(target=updateUi, args=(window, depth_image_8U, color_image, fps, totalseconds)).start()
            #updateUi(window, depth_image_8U, color_image, fps, totalseconds)
            if not data_queue.empty():
                worker.progress.emit(depth_image_8U, color_image, data_queue.get(), fps, totalseconds)
            else:
                worker.progress.emit(depth_image_8U, color_image, None, fps, totalseconds)
            #updateuiend = datetime.datetime.now()    # Wait for the reader and writer threads to exit
            #updateuitimes.append((updateuiend - updateuistart).total_seconds())

    finally:
        print("Waiting for writing process to finish...")

        # Stop streaming
        pipeline.stop()

        colorwqueue.put("DONE")
        depthwqueue.put("DONE")
        info_queue.put("DONE")
        writethread.join()
        audio_write_thread.join()

        first_wave_file.writeframes(b''.join(first_data))
        first_wave_file.close()

        second_wave_file.writeframes(b''.join(second_data))
        second_wave_file.close()
        #write(first_path, RATE, np.asarray(first_data).astype(np.int16))

        print("FPS: {:.2f}".format(fps))
        print("Operation Times:")
        captureavg = sum(capturetimes) / len(capturetimes)
        alignavg = sum(aligntimes) / len(aligntimes)
        numpyavg = sum(numpytimes) / len(numpytimes)
        #writeavg = sum(writetimes) / len(writetimes)
        #updateuiavg = sum(updateuitimes) / len(updateuitimes)

        print("Capture {} ms".format(captureavg * 1000))
        print("Align: {} ms".format(alignavg * 1000))
        print("Numpy: {} ms".format(numpyavg * 1000))
        #print("Write: {} ms".format(writeavg * 1000))
        #print("Update UI: {} ms".format(updateuiavg * 1000))



