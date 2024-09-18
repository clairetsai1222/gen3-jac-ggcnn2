'''
For learning purpose, I create 4 versions of the code for different progress completion.

Level 02: Utilize yolo model to predict grasped object and get the chosen item's grasp pose.

execution summary:
1. Load the yolo model and the ggcnn model.
2. Configure the realsense camera.
3. Start the realsense camera and get the depth and color images.
4. Preprocess the depth image with yolo outcome and use ggcnn model to predict the grasped object.
'''

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import datetime

from models.ggcnn2 import GGCNN2
from utils.calibrate import align_depth_color
from utils.dataset_processing import grasp, grocess_output, take_place_utils
from utils.calibrate import statical_camera_info

# 导入YOLO模型相关的库
from ultralytics import YOLO

# 加载相机内参
depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale = statical_camera_info.get_camera_intrinsics()

# 创建RealSense管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

print("Realsense Camera is started.")

# Load a model
yolo_model = YOLO("yolov8n.yaml")  # build a new model from scratch
yolo_model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# 加载GGCNN模型
ggcnn_model = GGCNN2()
ggcnn_model.load_state_dict(torch.load('./ggcnn2_weights_jacquard/epoch_100_iou_97_statedict.pt'))

expecting_detected_object = input("Please input the label of the object you want to detect: ")

print("Start detecting...")

stop_flag = True
try:
    while stop_flag:
        # 等待下一帧
        frames = pipeline.wait_for_frames()  
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 调整深度图像和彩色图像尺寸为704x1280
        target_size = (704, 1280)
        resized_depth_image, resized_color_image = take_place_utils.resize_images(depth_image, color_image, target_size)
        

        # 对齐深度图像和彩色图像
        dc_align_extrinsics = statical_camera_info.align_depth_color_extrinsics()
        aligned_depth_image = align_depth_color.align_images(resized_depth_image, resized_color_image, depth_intrinsic_matrix, color_intrinsic_matrix, dc_align_extrinsics)

        # 使用YOLO模型检测物体
        with torch.no_grad():
            color_image_tensor = torch.from_numpy(resized_color_image.transpose(2, 0, 1)).unsqueeze(0).float()
            color_image_tensor /= 255.0  # 归一化处理
            detections = yolo_model(color_image_tensor)
        
        print("-----------")
        print(detections)
        
        print("-----------")
        # 提取检测结果
        detections = detections[0].boxes.data.cpu().numpy()
        print(detections)
        print("-----------")

        # 我们只关心自己指定的物体，所以只需要检测指定的物体
        if len(detections) > 0:
            for detection in detections:
                label = yolo_model.names[int(detection[5])]  # 获取标签
                if label == expecting_detected_object: 
                    x1, y1, x2, y2 = detection[:4]
                    object_depth_image = resized_depth_image[int(y1):int(y2), int(x1):int(x2)]

                    # 使用GGCNN模型预测抓取点
                    with torch.no_grad():
                        depthT = torch.from_numpy(object_depth_image.reshape(1, 1, y2-y1, x2-x1).astype(np.float32))
                        grasp_imgs = ggcnn_model(depthT)

                    q_img, ang_img, width_img = grocess_output.post_process_output(q_img=grasp_imgs[0], cos_img=grasp_imgs[1], sin_img=grasp_imgs[2], width_img=grasp_imgs[3])
                    grasps = grasp.detect_grasps(q_img=q_img, ang_img=ang_img, width_img=width_img, no_grasps=1)

                    for grasp_objects in grasps:
                        horizon_angle = grasp_objects.as_gr.angle
                        grasp_point = grasp_objects.as_gr.as_grasp
                        rectangle_center = grasp_objects.as_gr.center
                        rectengle_length = grasp_objects.as_gr.length
                        rectengle_width = grasp_objects.as_gr.width
                        polygon_points = grasp_objects.as_gr.polygon_coords

                        # 将抓取点转换为相机坐标系
                        grasp_point_camera_frame = (rectangle_center[0] + x1, rectangle_center[1] + y1)

                        # 在深度图像和彩色图像上绘制抓取点
                        x, y = grasp_point_camera_frame
                        depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)
                        cv2.circle(depth_image_color, (x, y), 5, (0, 255, 0), -1)
                        cv2.circle(resized_color_image, (x, y), 5, (0, 255, 0), -1)

                        cv2.imshow('Depth Image with Grasp Point', depth_image_color)
                        cv2.imshow('Color Image with Grasp Point', resized_color_image)

                        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        depth_image_path = f'./grasp_output/yolo_grasp_image/depth_image_{current_time}.png'
                        cv2.imwrite(depth_image_path, depth_image_color)
                        color_image_path = f'./grasp_output/yolo_grasp_image/color_image_{current_time}.png'
                        cv2.imwrite(color_image_path, color_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
