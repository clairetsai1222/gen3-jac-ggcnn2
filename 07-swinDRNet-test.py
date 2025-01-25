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
# import torch
# import datetime
import time

# from models.ggcnn2 import GGCNN2
from PIL import Image
# from utils.dataset_processing import grasp, grocess_output, take_place_utils
from utils.calibrate import statical_camera_info
from utils.d435i_depth_detect import realsense_depth
from utils.log import save_log
# import utils.action.meta_action as ma

# YOLO
# from utils.yolo import object_detection
# from ultralytics import YOLO

# Grounding Dino
# from utils.dino import object_detection
# SwinDRNet
from utils.depth_repair import repair

# 加载相机内参
_, depth_scale, color_coefficients = statical_camera_info.get_camera_intrinsics()

# 创建RealSense管道
# Initialize Camera Intel Realsense
dc = realsense_depth.DepthCamera()

TS_hand_eye_file = "./TS_hand-eye/T_cam2base_calibrateHandEye_0.npz"

# create depth repair pipeline
repairPipeline = repair.SwinDRNetPipeline("./SwinDRNet/models/model.pth")

stop_flag = True
try:
    while stop_flag:
        ret, depth_image, color_image, depth_intrinsic, color_intrinsic = dc.get_frame()
        
        if not ret:
            continue

        # repair
        repaired_depth_image = repairPipeline.inference(np.array(color_image), np.array(depth_image), 2)
        # post process
        factor = 7
        repaired_depth_scaled_img = cv2.applyColorMap((repaired_depth_image/factor).astype(np.uint8), cv2.COLORMAP_JET)
        depth_scaled_img = cv2.applyColorMap((depth_image/factor).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('color_image', color_image)
        to_save = {\
                    "origina_color_image": Image.fromarray(color_image).convert("RGB"),\
                    "original_depth_image": Image.fromarray(depth_scaled_img).convert("RGB"),\
                    "repaired_depth_image": Image.fromarray(repaired_depth_scaled_img).convert("RGB"),\
                }
        time.sleep(3)

        if cv2.waitKey(0) & 0xFF == ord('g'):
            stop_flag = False
            save_log.Save(to_save)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            stop_flag = False

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    dc.release()
    cv2.destroyAllWindows()
