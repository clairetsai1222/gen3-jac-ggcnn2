'''
For learing purpose, I create 4 versions of the code for different progress completion.

Version 01: Using the grasp-cnn net model to predict the grasp point in the camera frame.

execuion summary:
- Get the camera intrinsics and depth scale from the camera.
- Load the model and set it to evaluation mode.
- Configure the Realsense camera and start it.
- Loop through frames and predict the grasp point using the model.
- Draw the grasp point on the depth and color images.
- Display the images and wait for a key press to exit.

'''


import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import datetime

from models.ggcnn import GGCNN
from models.ggcnn2 import GGCNN2
from ggcnn_torch import predict, process_depth_image

from utils.dataset_processing import grasp, grocess_output


# 将相机坐标转换为机械臂坐标系
def camera_to_robot_frame(camera_point, T_cam2gripper, intrinsics, depth_scale):
    # 相机坐标系下的点
    u, v, depth = camera_point

    # 反投影到相机坐标系
    Z = depth * depth_scale
    X = (u - intrinsics.ppx) * Z / intrinsics.fx
    Y = (v - intrinsics.ppy) * Z / intrinsics.fy

    # 相机坐标系下的点
    point_camera = np.array([X, Y, Z, 1])

    # 转换到机械臂坐标系
    point_robot = np.dot(T_cam2gripper, point_camera)

    return point_robot[:3]


# 获取我相机内参, 如果更换相机，需要重新获取内参，realsense d435i相机的获取方式在gen3_utils/d435i_intrinsics.py中
def get_camera_intrinsics():
    # 深度相机的内参矩阵
    depth_intrinsic_matrix = np.array([
        [427.52703857,   0.          , 426.38412476],
        [  0.          , 427.52703857, 237.69470215],
        [  0.          ,   0.        ,   1.        ]
    ])
    
    # 颜色相机的内参矩阵
    color_intrinsic_matrix = np.array([
        [910.4362793 ,   0.        , 647.40075684],
        [  0.        , 910.06066895, 364.79605103],
        [  0.        ,   0.        ,   1.        ]
    ])
    
    # 深度比例因子
    depth_scale = 999.999952502551
    
    return depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale

# 主函数

# 加载相机内参
depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale = get_camera_intrinsics()

# 配置Realsense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 启动摄像头
pipeline.start(config)

stop_flag = True
try:
    while stop_flag:
        # 获取图像帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 将图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        
        color_image = np.asanyarray(color_frame.get_data())

        # 预处理深度图像（假设模型需要归一化）
        # depth_image_normalized = depth_image.astype(np.float32) / 65535.0  # 归一化到[0, 1]
        # depth_image_tensor = torch.tensor(depth_image_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        model = GGCNN()
        # model.load_state_dict(torch.load('ggcnn_weights_jacquard/epoch_100_iou_97_statedict.pt'))
        model.load_state_dict(torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'))

        # 使用模型预测抓取点
        with torch.no_grad():
            depthT = torch.from_numpy(depth_image.reshape(1, 1, 848, 480).astype(np.float32))
            grasp_imgs = model(depthT)

        q_img, ang_img, width_img = grocess_output.post_process_output(q_img = grasp_imgs[0], cos_img = grasp_imgs[1], sin_img = grasp_imgs[2], width_img=grasp_imgs[3])
        grasps = grasp.detect_grasps (q_img = q_img, ang_img=ang_img, width_img=width_img, no_grasps=1)

        # 模型输出一个抓取的框架，需要将其转换为机械臂坐标系下的抓取点
        for grasp_objects in grasps:
            horizon_angle = grasp_objects.as_gr.angle
            grasp_point = grasp_objects.as_gr.as_grasp
            rectangle_center = grasp_objects.as_gr.center
            rectengle_length = grasp_objects.as_gr.length
            rectengle_width = grasp_objects.as_gr.width
            polygon_points = grasp_objects.as_gr.polygon_coords
            print("grap point in camera frame:", rectangle_center)

        # # 在深度图像和彩色图像上绘制抓取点
        x, y = rectangle_center[0], rectangle_center[1]
        depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.circle(depth_image_color, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Depth Image with Grasp Point', depth_image_color)
        cv2.imshow('Color Image with Grasp Point', color_image)

        # 获取当前系统时间并格式化为字符串
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存深度图像
        depth_image_path = f'./grasp_output/ori_grasp_image/depth_image_{current_time}.png'  # 指定保存路径
        cv2.imwrite(depth_image_path, depth_image_color)

        # 保存彩色图像
        color_image_path = f'./grasp_output/ori_grasp_image/color_image_{current_time}.png'  # 指定保存路径
        cv2.imwrite(color_image_path, color_image)
        
        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止摄像头并关闭所有窗口
    pipeline.stop()
    cv2.destroyAllWindows()

