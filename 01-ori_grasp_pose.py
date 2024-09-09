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
from models.ggcnn2 import GGCNN2

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
# 加载模型
# Option 1) Load the model directly.
# (this may print warning based on the installed version of python)
model = torch.load('ggcnn_weights_jacquard/epoch_100_iou_0.97')

# Option 2) Instantiate a model and load the weights.
model = GGCNN2()
model.load_state_dict(torch.load('ggcnn_weights_jacquard/epoch_100_iou_0.97_statedict.pt'))

# 配置Realsense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 启动摄像头
pipeline.start(config)

try:
    while True:
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
        depth_image_normalized = depth_image.astype(np.float32) / 65535.0  # 归一化到[0, 1]
        depth_image_tensor = torch.tensor(depth_image_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 增加batch和channel维度

        # 使用模型预测抓取点
        with torch.no_grad():
            grasp_point = model(depth_image_tensor)

        # 假设模型输出的是抓取点的坐标 (x, y)
        grasp_point = grasp_point.squeeze().numpy()
        x, y = int(grasp_point[0]), int(grasp_point[1])

        # 在深度图像和彩色图像上绘制抓取点
        depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.circle(depth_image_color, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)

        # 显示图像
        cv2.imshow('Depth Image with Grasp Point', depth_image_color)
        cv2.imshow('Color Image with Grasp Point', color_image)

        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止摄像头并关闭所有窗口
    pipeline.stop()
    cv2.destroyAllWindows()

