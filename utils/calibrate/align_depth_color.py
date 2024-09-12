import cv2
import numpy as np
import pyrealsense2 as rs
import os

# from statical_camera_info import get_camera_intrinsics, align_depth_color_extrinsics
from utils.calibrate import statical_camera_info




def align_images(depth_image, color_image, depth_intrinsics, color_intrinsics, align_extrinsics):
    # 获取深度图像和彩色图像的尺寸
    depth_height, depth_width = depth_image.shape[:2]
    color_height, color_width = color_image.shape[:2]

    # 创建一个空的输出图像
    aligned_depth_image = np.zeros((color_height, color_width), dtype=np.float32)

    # 遍历深度图像的每个像素
    for y in range(depth_height):
        for x in range(depth_width):
            # 获取深度值
            depth = depth_image[y, x]
            if depth == 0:  # 忽略无效深度值
                continue

            # 将深度图像中的像素坐标转换为3D点
            z = depth
            x_3d = (x - depth_intrinsics[0, 2]) * z / depth_intrinsics[0, 0]
            y_3d = (y - depth_intrinsics[1, 2]) * z / depth_intrinsics[1, 1]

            # 将3D点转换到彩色相机坐标系
            point_3d = np.array([x_3d, y_3d, z, 1])
            point_3d_color = np.dot(align_extrinsics, point_3d)

            # 将彩色相机坐标系中的点投影到彩色图像上
            x_color = int((point_3d_color[0] / point_3d_color[2]) * color_intrinsics[0, 0] + color_intrinsics[0, 2])
            y_color = int((point_3d_color[1] / point_3d_color[2]) * color_intrinsics[1, 1] + color_intrinsics[1, 2])

            # 检查投影点是否在彩色图像范围内
            if 0 <= x_color < color_width and 0 <= y_color < color_height:
                aligned_depth_image[y_color, x_color] = depth

    return aligned_depth_image

def depth_to_pseudo_color(depth_image, max_depth=10000):
    # 将深度图像归一化到 [0, 1] 范围
    normalized_depth = depth_image / max_depth
    normalized_depth = np.clip(normalized_depth, 0, 1)
    
    # 将归一化的深度图像转换为伪彩色图像
    pseudo_color_depth = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return pseudo_color_depth

def overlay_images(color_image, pseudo_color_depth, alpha=0.5):
    # 将彩色图像和伪彩色深度图像叠加
    overlay = cv2.addWeighted(color_image, 1 - alpha, pseudo_color_depth, alpha, 0)
    return overlay


def capture_images_from_realsense():
    # 创建管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置深度和彩色流
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 开始流
    pipeline.start(config)

    print("按下 's' 键保存图像")
    try:
         # 等待按键
        while True:
            # 等待一对匹配的帧
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("无法获取深度或彩色帧")
                return

            # 将帧转换为OpenCV图像
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 显示彩色图像
            cv2.imshow('Color Image', color_image)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF


            if key == ord('s'):
                # 创建存储文件夹
                if not os.path.exists('align_depth_color_example_image'):
                    os.makedirs('align_depth_color_example_image')

                # 保存图像
                cv2.imwrite('align_depth_color_example_image/depth_image.png', depth_image)
                cv2.imwrite('align_depth_color_example_image/color_image.png', color_image)

                print("图像已成功捕获并保存")
                break
            elif key == 27:  # ESC键退出
                print("取消保存图像")
                break

    finally:
        # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()




# 示例使用

# # 捕获图像，如果不想重新捕获，可以注释掉这一步
# # capture_images_from_realsense()

# # 加载图像
# depth_image = cv2.imread('./align_depth_color_example_image/depth_image.png', cv2.IMREAD_ANYDEPTH)
# color_image = cv2.imread('./align_depth_color_example_image/color_image.png')

# # 加载相机内参和对齐深度图像和彩色图像的外参矩阵
# depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale = statical_camera_info.get_camera_intrinsics()
# dc_align_extrinsics = statical_camera_info.align_depth_color_extrinsics()

# aligned_depth_image = align_images(depth_image, color_image, depth_intrinsic_matrix, color_intrinsic_matrix, dc_align_extrinsics)

# # 将深度图像转换为伪彩色图像
# pseudo_color_depth = depth_to_pseudo_color(aligned_depth_image)

# # 将彩色图像和伪彩色深度图像叠加
# overlay_image = overlay_images(color_image, pseudo_color_depth)

# # 显示叠加后的图像
# cv2.imshow('Overlay Image', overlay_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()