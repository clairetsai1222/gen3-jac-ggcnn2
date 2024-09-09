'''

'''

import pyrealsense2 as rs
import numpy as np
import cv2
from gen3_utils import object_yolo, extrinsics_calibration

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

# 假设你有一个控制Kinova Gen3机械臂的函数
def control_kinova(grasp_point):
    # 控制机械臂移动到抓取点
    pass


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

    

# 进行手眼标定
def load_calibrate_hand_eye():


# 从bounding box中求取抓取点
def load_bbox_grasp_point(depth_image, bbox):
    

# 控制机械臂
def control_kinova(grasp_point):
    # 控制机械臂移动到抓取点
    pass

# 主函数
def __main__():
    # 配置Realsense管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 获取相机内参
    intrinsics, depth_scale = get_camera_intrinsics()

    # 进行手眼标定
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    T_cam2gripper = load_calibrate_hand_eye

    try:
        while True:
            # 等待下一帧
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 将图像转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 进行物体检测
            object_name = 'your_object_name'  # 指定物体名称

            # 从bounding box中求取抓取点
            grasp_point = get_grasp_point(depth_image, bbox)

            # 显示抓取点
            cv2.circle(color_image, (grasp_point[0], grasp_point[1]), 5, (0, 0, 255), -1)

            # 将相机坐标转换为机械臂坐标系
            robot_grasp_point = camera_to_robot_frame(grasp_point, T_cam2gripper, intrinsics, depth_scale)

            # 控制机械臂
            control_kinova(robot_grasp_point)

            # 显示图像
            cv2.imshow('RealSense Color', color_image)
            cv2.imshow('RealSense Depth', cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    __main__()

