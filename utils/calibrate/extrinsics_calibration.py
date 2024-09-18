import cv2
import numpy as np
import pyrealsense2 as rs
import os
import datetime
import statical_camera_info
# import sys
# import time
# import threading
# import utilities

# from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
# from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

# from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2


# def cartesian_arm_position(base, base_cyclic):
#     '''
#     Arg:
#     base: an instance of BaseClient 基础客户端
#     base_cyclic: an instance of BaseCyclicClient 循环基础客户端
#     '''
    
#     print("Geting the current cartesian position of the arm ...")
#     action = Base_pb2.Action()
#     action.name = "Example Cartesian action movement"
#     action.application_data = "" 

#     feedback = base_cyclic.RefreshFeedback() # 从base_cyclic客户端获取当前的反馈信息

#     cartesian_pose = action.reach_pose.target_pose
#     cartesian_pose.x = feedback.base.tool_pose_x     # (meters)
#     cartesian_pose.y = feedback.base.tool_pose_y     # (meters)
#     cartesian_pose.z = feedback.base.tool_pose_z     # (meters)
#     cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)夹爪角度：+往下；-往上
#     cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)夹爪角度：+逆时针；-顺时针
#     cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)夹爪角度：+左转；-右转
#     arm_position = np.array([cartesian_pose.x, cartesian_pose.y, cartesian_pose.z])  # 机械臂末端执行器的位置 (x, y, z)
#     arm_orientation = np.array([cartesian_pose.theta_x, cartesian_pose.theta_y, cartesian_pose.theta_z])  # 机械臂末端执行器的旋转角度 (roll, pitch, yaw)

#     R, _ = cv2.Rodrigues(arm_orientation)
#     translation = arm_position.reshape(3, 1)
#     transform_matrix = np.hstack((R, translation))
#     transform_matrix = np.vstack((transform_matrix, [0, 0, 0, 1]))

#     return transform_matrix

def get_ex_calibrate(image_folder, chessboard_size=(8, 11)):
    """获取Realsense D435i相机的彩色和深度图像，并显示和保存"""
    # 创建管道以获取Realsense相机数据
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 启动管道
    pipeline.start(config)

    try:
        while True:
            # 等待帧的到来
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 在彩色图像上进行棋盘格角点检测
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            conner_color_image = color_image.copy()

            if ret:
                # 如果检测到角点，则在图像上绘制角点
                cv2.drawChessboardCorners(conner_color_image, chessboard_size, corners, ret)

            # 显示带有角点的图像
            cv2.imshow('Detected Chessboard Corners', conner_color_image)

            key = cv2.waitKey(1)

            # 按s键保存图像
            if key & 0xFF == ord('s') and ret:
                # 获取当前时间并格式化为字符串
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # 生成带有时间戳的文件名
                color_filename = os.path.join(image_folder, f"calibration_image_{current_time}.png")
                depth_filename = os.path.join(image_folder, f"depth_image_{current_time}.png")

                cv2.imwrite(color_filename, color_image)

                print(f"图像已保存: {color_filename} 和 {depth_filename}")

                # 计算相机坐标系到机械臂坐标系的变换矩阵
                extrinsic_matrix = calculate_extrinsics(chessboard_size, image_folder)

            # 按q键退出
            elif key & 0xFF == ord('q'):
                break

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return extrinsic_matrix


def calculate_extrinsics(chessboard_size, image_folder):
    """根据棋盘图案的检测结果计算相机外参"""
    # 生成棋盘格的3D点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D点在世界坐标系中
    imgpoints = []  # 2D点在图像平面中

    # 读取图像并进行角点检测
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 找到棋盘角点
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        else:
            print(f"图像{filename}读取失败")

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("本次存储图像未找到任何角点，无法进行相机标定。")
    # 相机内参与畸变系数的计算
    # 加载相机内参
    depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale = statical_camera_info.get_camera_intrinsics()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], color_intrinsic_matrix, None)

    for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        T = tvecs[i]
        # 创建一个 4x4 的外参矩阵
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = R  # 将旋转矩阵放入左上角
        extrinsic_matrix[:3, 3] = T.flatten()  # 将平移向量放入最后一列
        extrinsic_matrix[3, 3] = 1  # 齐次坐标的最后一行应为 [0, 0, 0, 1]
    print("外参矩阵:\n", extrinsic_matrix)

    return extrinsic_matrix


def main():
    chessboard_size = (7, 10)  # 棋盘格尺寸
    image_folder = './ex_calibration_images'

    # 获取外参标定所需的图像
    extrinsic_matrix = get_ex_calibrate(image_folder, chessboard_size)

    # # 保存外参矩阵
    # np.save('extrinsic_matrix.npy', extrinsic_matrix)

if __name__ == "__main__":
    main()