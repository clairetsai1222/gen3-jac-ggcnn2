'''
For learning purpose, I create 4 versions of the code for different progress completion.

Level 03: 
I have gotten specific object's 2D grasp point on depth image. 
Now I want to get the 3D grasp point on the real world. 
I need to use the camera intrinsic parameters and the grasp point on the 2D image to get the 3D grasp point.
Also, I will calibrate the camera with robot arm to get the extrinsic matrix. 
And utilize the extrinsic matrix to get the 3D grasp point in robot base frame.

execution summary:
1. use the camera intrinsic parameters to get the 3D grasp point on the real world.
2. calibrate the camera with robot arm to get the extrinsic matrix.
3. utilize the extrinsic matrix to get the 3D grasp point in robot base frame.
'''

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import datetime
import time

from models.ggcnn2 import GGCNN2
from utils.dataset_processing import grasp, grocess_output, take_place_utils
from utils.calibrate import statical_camera_info
from utils.d435i_depth_detect import realsense_depth
# import gen3_move_cartesiancopy as move

# YOLO
# from utils.yolo import object_detection
# from ultralytics import YOLO

# Grounding Dino
from utils.dino import object_detection
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate


# 加载相机内参
# _, depth_scale, color_coefficients = statical_camera_info.get_camera_intrinsics()

# 创建RealSense管道
# Initialize Camera Intel Realsense
# dc = realsense_depth.DepthCamera()

# TS_hand_eye_file = "./TS_hand-eye/T_cam2base_calibrateHandEye_0.npz"
expecting_detected_object = input("Please input the label of the object you want to detect: ")
image_source, image = load_image("dog.jpeg")
# resized_depth_image, resized_color_image = take_place_utils.resize_images(im, im, (704, 1280))
detect_result = object_detection.ObjectDetection(detect_object=expecting_detected_object, color_image=image, detpth_image=image, color_intrinsics=None)
objects_dict, object_keys = detect_result.get_results()
print(detect_result.object_keys)