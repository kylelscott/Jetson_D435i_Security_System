import pyrealsense2 as rs
import numpy as np
import cv2
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        start = time.time()
        color_image = np.asanyarray(color_frame.get_data())
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        (h,w) = color_image.shape[:2]
        cv2.namedWindow("RealSense Stream", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Realsense", color_image)
        cv2.waitKey(1)

        end = time.time()
        sec = end-start
        fps = 1/sec
        logging.info("FPS: {}".format(fps))
finally:
    pipeline.stop()