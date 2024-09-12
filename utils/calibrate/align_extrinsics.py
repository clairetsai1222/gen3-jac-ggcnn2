import cv2
import numpy as np
import pyrealsense2 as rs
import os

def calibrate_camera_and_get_extrinsics(chessboard_size, image_folder):
    # 准备对象点，如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # 存储对象点和图像点的数组
    objpoints = []  # 3d点在世界坐标系中
    imgpoints = []  # 2d点在图像平面中

    # 创建RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 启动管道
    pipeline.start(config)

    # 创建保存图像的文件夹
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_count = 0

    try:
        while True:
            # 等待下一帧
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 将帧转换为OpenCV格式
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 显示彩色图像
            cv2.imshow('Color Image', color_image)

            # 等待按键
            key = cv2.waitKey(1) & 0xFF

            # 按下's'键保存图像
            if key == ord('s'):
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # 查找棋盘格角点
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                print(f"Found corners: {ret}")

                # 如果找到，添加对象点和图像点
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)

                    # 保存图像
                    image_path = os.path.join(image_folder, f'image_{image_count}.jpg')
                    cv2.imwrite(image_path, color_image)
                    print(f"Saved image: {image_path}")
                    image_count += 1

            # 按下'q'键退出
            if key == ord('q'):
                break

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

    # 标定相机
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 获取外参矩阵
    extrinsics = np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))
    extrinsics = np.vstack((extrinsics, [0, 0, 0, 1]))

    return extrinsics

# 使用示例
chessboard_size = (17, 12)  # 例如，9x6的棋盘格
image_folder = 'align_calibration_images'
extrinsics_matrix = calibrate_camera_and_get_extrinsics(chessboard_size, image_folder)
print("align extrinsics matrix:\n", extrinsics_matrix)
