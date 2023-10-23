import time
import datetime
import os
import threading
import wave
from queue import Queue
import multiprocessing

import pyrealsense2.pyrealsense2 as rs
import pyaudio

import numpy as np
import cv2

import h5py

def save_intrinsics(f, group_name, intrin):
    f.create_dataset(group_name + '/coeffs', data = intrin.coeffs)
    f.create_dataset(group_name + '/fx', data = intrin.fx)
    f.create_dataset(group_name + '/fy', data = intrin.fy)
    f.create_dataset(group_name + '/width', data = intrin.width)
    f.create_dataset(group_name + '/height', data = intrin.height)
    #f.create_dataset(group_name + '/model', data = intrin.model)
    f.create_dataset(group_name + '/ppx', data = intrin.ppx)
    f.create_dataset(group_name + '/ppy', data = intrin.ppy)

def save_extrinsics(f, group_name, extrin):
    f.create_dataset(group_name + '/rotation', data = extrin.rotation)
    f.create_dataset(group_name + '/translation', data = extrin.translation)

def process_video_queue(color_queue, depth_queue, dims, dateandtime):
    output_dir = os.path.join(os.getcwd(), "videos")

    # setup video save path
    color_path = os.path.join(output_dir, "{}_rgb.avi".format(dateandtime))
    depth_path = os.path.join(output_dir, "{}_depth.avi".format(dateandtime))

    print("[INFO] Saving video of dimensions {} x {}".format(dims[0], dims[1]))
    print("[INFO] Saving color stream to: {}".format(color_path))
    print("[INFO] Saving depth stream to: {}".format(depth_path))

    # init writers
    color_writer = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*"XVID"), 30, dims, 1)
    depth_writer = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*"DIVX"), 30, dims, 0)

    # queue process loop
    while True:
        if color_queue.empty() and depth_queue.empty():
            continue
        color_data = color_queue.get()
        depth_data = depth_queue.get()
        if type(color_data) == str and type(depth_data) == str:
            break

        if type(depth_data) != str:
            depth_writer.write(depth_data)
        if type(color_data) != str:
            color_writer.write(color_data)


def progress_callback(progress):
    print(f'\rProgress  {progress}% ... ', end ="\r")


def config_video(worker, date_and_time):

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)

    # init profile
    try:
        pipeline_profile = config.resolve(pipeline_wrapper)
    except:
        print("ERROR: No device connected")
        worker.window.errorLabel.setHidden(False)
        return None, None, None, None, None

    device = pipeline_profile.get_device()

    # check for depth and rgb sensor
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The program requires Depth camera with Color sensor")
        return None, None, None, None, None

    dims = worker.window.dim

    # enable stream
    config.enable_stream(rs.stream.depth, dims[0], dims[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, dims[0], dims[1], rs.format.bgr8, 30)

    # align stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline.start(config)

    # init queue
    color_queue = multiprocessing.Queue()
    depth_queue = multiprocessing.Queue()

    # init write thread
    write_thread = multiprocessing.Process(target=process_video_queue, args=(color_queue, depth_queue, dims, date_and_time))
    write_thread.start()

    return pipeline, align, color_queue, depth_queue, write_thread

def process_audio_queue(worker, first_data, second_data, first_wave_file, second_wave_file, info_queue, data_queue):
    global frame_captured, num_samples
    num_samples = 0

    FORMAT = pyaudio.paInt16  # We use 16bit format per sample
    CHANNELS = 2
    RATE = worker.window.fs

    print("Sampling rate: {} Hz".format(RATE))

    CHUNK = 1024*4
    DEVICE_FIRST = 0
    DEVICE_SECOND = 0

    audio = pyaudio.PyAudio()

    info = audio.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    temp_iter = 0

    selected_index_first = worker.window.audioComboBox.currentIndex()
    print("[DEBUG] Selected index first: {}".format(selected_index_first))

    selected_index_second = worker.window.audioComboBox_2.currentIndex()
    print("[DEBUG] Selected index second: {}".format(selected_index_second))

    second_audio_disabled = worker.window.disableSecondBox.isChecked()

    for i in range(0, num_devices):
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
                              input_device_index=DEVICE_FIRST
                              )
    if not second_audio_disabled:
        stream_second = audio.open(format=FORMAT,
                                   channels=CHANNELS,
                                   rate=RATE,
                                   input=True,
                                   output=False,
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
            if frame_captured:
                audio_first = stream_first.read(CHUNK)
                data_queue.put(audio_first)
                if not second_audio_disabled:
                    audio_second = stream_second.read(CHUNK)

                first_data.append(audio_first)
                if not second_audio_disabled:
                    second_data.append(audio_second)

                num_samples += CHUNK
            else:
                time.sleep(0.01)

    finally:
        stream_first.stop_stream()
        stream_first.close()

        if not second_audio_disabled:
            stream_second.stop_stream()
            stream_second.close()

        audio.terminate()


def recording(worker):

    global frame_captured, num_samples
    frame_captured = False

    pipeline = None
    align = None
    color_queue = None
    depth_queue = None
    write_thread = None

    disable_video = worker.window.disableVideoBox.isChecked()

    date_and_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    if not disable_video:
        pipeline, align, color_queue, depth_queue, write_thread = config_video(worker, date_and_time)
        if pipeline is None:
            worker.window.start_clicked()
            return

    # create output folder if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "videos")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f"Folder {output_dir} created successfully!")
    else:
        print(f"Folder {output_dir} already exists.")

    second_audio_disabled = worker.window.disableSecondBox.isChecked()

    # setup audio files path
    first_path = os.path.join(output_dir, "{}_first.wav".format(date_and_time))
    second_path = os.path.join(output_dir, "{}_second.wav".format(date_and_time))

    first_wave_file = wave.open(first_path, 'wb')
    print("Saving first wave to file: {}".format(first_path))

    if not second_audio_disabled:
        second_wave_file = wave.open(second_path, 'wb')
        print("Saving second wave to file: {}".format(second_path))
    else:
        second_wave_file = None

    frame_num = 0
    fps = 0
    start_time = datetime.datetime.now()
    fps_time = start_time

    capture_times = []
    numpy_times = []

    info_queue = Queue()
    data_queue = Queue()
    first_data = []
    second_data = []

    color_image = None
    depth_image_8U = None

    # get & save camera parameters
    if not disable_video:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        with h5py.File(output_dir + '/{}_param.h5'.format(date_and_time), mode='w') as f:
            save_intrinsics(f, '/camera_param/depth_intrin', depth_intrin)
            save_intrinsics(f, '/camera_param/color_intrin', color_intrin)
            save_extrinsics(f, '/camera_param/depth_to_color_extrin', depth_to_color_extrin)

    audio_write_thread = threading.Thread(target=process_audio_queue, args=(worker, first_data, second_data,
                                                                            first_wave_file, second_wave_file,
                                                                            info_queue, data_queue))

    audio_write_thread.start()

    fp_sync = open(output_dir + '/{}_sync.csv'.format(date_and_time), 'w')

    leftover_time = 0.0

    try:
        while worker.window.isRecording:
            if not disable_video:
                if (datetime.datetime.now() - fps_time).total_seconds() > 5:
                    frame_num = 0
                    fps_time = datetime.datetime.now()

                # Wait for a coherent pair of frames: depth and color
                capture_start = datetime.datetime.now()

                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                capture_end = datetime.datetime.now()

                if not depth_frame or not color_frame:
                    leftover_time += (capture_end - capture_start).total_seconds()
                    continue
                capture_times.append((capture_end - capture_start).total_seconds() + leftover_time)

                if not frame_captured:
                    frame_captured = True
                fp_sync.write("{:.0f}\n".format(num_samples))

                # Filters (comment in and out for best effect)
                # depth_frame = rs.hole_filling_filter().process(depth_frame)
                # depth_frame = colorizer.colorize(depth_frame)

                numpy_start = datetime.datetime.now()
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_image_8U = cv2.convertScaleAbs(depth_image, alpha=0.03)

                # Built in opencv histogram equalization
                if worker.window.histogramBox.currentIndex() == 2:
                    depth_image_8U = cv2.equalizeHist(depth_image_8U)

                numpy_end = datetime.datetime.now()
                numpy_times.append((numpy_end - numpy_start).total_seconds())

                color_queue.put(color_image)
                depth_queue.put(depth_image_8U)

                frame_num += 1
                current_time = datetime.datetime.now()
                fps_seconds = (current_time - fps_time).total_seconds() + leftover_time
                fps = frame_num / fps_seconds
                leftover_time = 0.0

            else:
                time.sleep(0.01)
                current_time = datetime.datetime.now()

            total_seconds = (current_time - start_time).total_seconds()

            if not data_queue.empty():
                worker.progress.emit(depth_image_8U, color_image, data_queue, fps, total_seconds)
            else:
                worker.progress.emit(depth_image_8U, color_image, None, fps, total_seconds)

    finally:
        print("Waiting for writing process to finish...")

        # Stop streaming
        if not disable_video:
            pipeline.stop()
            color_queue.put("DONE")
            depth_queue.put("DONE")

        info_queue.put("DONE")
        if not disable_video:
            write_thread.join()
            color_queue.close()
            depth_queue.close()

        audio_write_thread.join()

        first_wave_file.writeframes(b''.join(first_data))
        first_wave_file.close()

        if not second_audio_disabled:
            second_wave_file.writeframes(b''.join(second_data))
            second_wave_file.close()

        print("Done!")
        fp_sync.close()

        if not disable_video:
            print("FPS: {:.2f}".format(fps))
            print("Operation Times:")
            capture_avg = sum(capture_times) / len(capture_times)
            numpy_avg = sum(numpy_times) / len(numpy_times)

            print("Capture and align: {} ms".format(capture_avg * 1000))
            print("Numpy: {} ms".format(numpy_avg * 1000))

