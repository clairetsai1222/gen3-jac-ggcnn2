import cv2
import numpy as np
import pyrealsense2 as rs

# 初始化Realsense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# 棋盘格参数
board_size = (9, 6)  # 棋盘格内角点数
square_size = 0.025  # 每个方格的边长（米）

# 存储棋盘格角点的世界坐标和图像坐标
object_points = []  # 世界坐标系中的3D点
image_points = []   # 图像坐标系中的2D点

# 生成棋盘格角点的世界坐标
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

# 采集数据
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(color_image, board_size, None)

    if ret:
        # 可视化角点
        cv2.drawChessboardCorners(color_image, board_size, corners, ret)
        cv2.imshow('Chessboard', color_image)

        # 等待按键输入
        key = cv2.waitKey(0)
        if key == ord('q'):  # 按下 'q' 键采集图像
            object_points.append(objp)
            image_points.append(corners)
            print("图像采集成功，按 'q' 继续采集，按 'c' 完成采集。")
        elif key == ord('c'):  # 按下 'c' 键完成采集
            break

# 停止相机
pipeline.stop()

# 获取相机内参
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, color_image.shape[::-1], None, None)

# 获取机械臂末端到相机的变换矩阵
# 这里假设你已经通过某种方式获取了机械臂末端到相机的变换矩阵
# 例如通过机械臂的API获取末端位置，并通过手眼标定获取相机到末端的变换矩阵
# 这里我们假设已经有一个变换矩阵 T_cam_to_end
T_cam_to_end = np.eye(4)  # 假设相机到机械臂末端的变换矩阵

# 手眼标定
# 这里我们假设已经通过某种方式获取了机械臂基座到末端的变换矩阵
# 例如通过机械臂的API获取基座到末端的变换矩阵
# 这里我们假设已经有一个变换矩阵 T_base_to_end
T_base_to_end = np.eye(4)  # 假设机械臂基座到末端的变换矩阵

# 计算相机到基座的变换矩阵
T_cam_to_base = np.dot(T_base_to_end, np.linalg.inv(T_cam_to_end))

# 打印结果
print("相机到基座的变换矩阵:")
print(T_cam_to_base)

# 关闭所有窗口
cv2.destroyAllWindows()
