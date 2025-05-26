import os
import argparse
import warnings
import json
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import logging
import datetime
import imutils
from twilio.rest import Client


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

#checking to see if Dropbox usage is enabled
if conf["use_dropbox"]:
    client = dropbox.Dropbox(conf["dropbox_access_token"])
    logging.info("Successfully connected to dropbox account")
    
#initialize some Twilio stuff
if conf["use_twilio"]:
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)
    logging.info("Up and Running")
    message = client.messages \
                .create(
                     body="<INSERT YOUR CUSTOM INITIALIZING MESSAGE HERE>",
                     from_="<INSERT YOUR TWILIO PHONE NUMBER HERE>",
                     to="<INSERT YOUR/THE PERSON TO BE NOTIFIED PHONE NUMBER HERE>"
                 )
    logging.info(message.sid)

avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0


while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    start = time.time()
    color_image = np.asanyarray(color_frame.get_data())
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    timestamp = datetime.datetime.now()
    text = "No occupants"

    frame = imutils.resize(color_image, width = 500)
    gray = frame[:,:,0]
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if avg is None:
        logging.info("starting background depiction...")
        avg = gray.copy().astype("float")
        logging.info("background depiction gathered")
        continue
        
    #get weighted avg between current and previous frame and compute difference between current frame and running avg
    cv2.accumulateWeighted(gray, avg, .5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    
    #threshold delta image, dilate threshold image, and find contours on threshold
    threshold = cv2.threshold(frameDelta,conf["delta_thresh"], 255,cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations=2)
    cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # loop over the contours
    for c in cnts:
        
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue        
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        text = "Occupied"
        
    #draw text and timestamp
    timestamp1 = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Room Status: {}".format(text), (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
    cv2.putText(frame, timestamp1, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1)
    
    if text == "Occupied":
        
        #check to see if enough time has passed btw uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            motionCounter += 1
            
            #check to see if number of frames with consistent motion is high enough
            if motionCounter >= conf["min_motion_frames"]:
                
                #check to see if dropbox should be used
                #if so do the following
                if conf["use_dropbox"]:
                    t = TempImage()
                    cv2.imwrite(t.path,frame)
                    
                    #upload image to dropbox and clear for new image upload
                    logging.info("[UPLOAD] {}".format(ts))
                    path = "/{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
                    client.files_upload(open(t.path, "rb").read(), path)
                    t.cleanup()
                    
                elif conf["use_twilio"]:
                    message = client.messages \
                .create(
                    body="<INSERT CUSTOM ALERT MESSAGE HERE>",
                    from_="<INSERT TWILIO PHONE NUMBER HERE>",
                    to="<INSERT RECIPIENT'S PHONE NUMBER HERE>"
                )
                    logging.info(message.sid)
                lastUploaded = timestamp
                motionCounter = 0
    else:
        motionCounter = 0
        
    
    #should frames be displayed on screen?
    if conf["show_video"]:
        #display feed
        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        
    #clear stream to prep for next frame
    


    (h,w) = color_image.shape[:2]
    cv2.namedWindow("RealSense Stream", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Realsense", color_image)
    cv2.waitKey(1)

    end = time.time()
    sec = end-start
    fps = 1/sec
    logging.info("FPS: {}".format(fps))

pipeline.stop()