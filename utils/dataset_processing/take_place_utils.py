import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt


'''
yolo的输入接口的图像尺寸要求为32的倍数，而Kinova的图像尺寸为720x1280，因此需要对图像进行压缩，压缩为704x1280的图像。
'''
def resize_images(depth_image, color_image, target_size):
    """
    将深度图像和彩色图像调整为目标尺寸，并保持数据存储格式相同。

    :param depth_image: 输入深度图像，numpy数组
    :param color_image: 输入彩色图像，numpy数组
    :param target_size: 目标尺寸，元组 (height, width)
    :return: 调整后的深度图像和彩色图像，numpy数组
    """
    resized_depth_image = cv2.resize(depth_image, target_size[::-1], interpolation=cv2.INTER_AREA)
    resized_color_image = cv2.resize(color_image, target_size[::-1], interpolation=cv2.INTER_AREA)
    return resized_depth_image, resized_color_image

# # 示例使用
# depth_image = np.random.randint(0, 255, (720, 1280), dtype=np.uint16)  # 示例深度图像
# color_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)  # 示例彩色图像
# target_size = (704, 1280)  # 目标尺寸
# resized_depth_image, resized_color_image = resize_images(depth_image, color_image, target_size)

# print(f"Original depth shape: {depth_image.shape}")
# print(f"Resized depth shape: {resized_depth_image.shape}")
# print(f"Original color shape: {color_image.shape}")
# print(f"Resized color shape: {resized_color_image.shape}")

def xywh_to_xyxy(xywh):
    """
    将 xywh 格式的边界框转换为 xyxy 格式。
    
    参数:
    xywh (list): 包含边界框的中心点坐标 (x_center, y_center) 和边界框的宽度 width 和高度 height。
    
    返回:
    list: 包含边界框的左上角坐标 (x_min, y_min) 和右下角坐标 (x_max, y_max)。
    """
    x_center, y_center, width, height = xywh
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]

def calculate_3d_grasp_point(grasp_point_2d, depth_image, depth_intrinsic_matrix, depth_scale):
    """
    计算3D抓取点。

    :param grasp_point_2d: 2D抓取点坐标 (x, y)
    :param depth_image: 深度图像
    :param depth_intrinsic_matrix: 深度相机的内参矩阵
    :param depth_scale: 深度图像的缩放因子
    :return: 3D抓取点 (x, y, z)
    """
    # 提取2D抓取点的坐标
    u, v = grasp_point_2d[1], grasp_point_2d[0]

    # 获取深度图像中对应点的深度值
    depth_value = depth_image[int(u), int(v)] * depth_scale

    # 如果深度值为0，表示该点没有有效的深度信息，返回None
    while depth_value <= 0:
        return None
    # 获取相机内参矩阵的参数
    fx = depth_intrinsic_matrix[0, 0]
    fy = depth_intrinsic_matrix[1, 1]
    cx = depth_intrinsic_matrix[0, 2]
    cy = depth_intrinsic_matrix[1, 2]

    # 计算3D抓取点
    x = (u - cx) * depth_value / fx
    y = (v - cy) * depth_value / fy
    return np.array([x, y, depth_value])


def colorize_depth_image(depth_image):
    """
    将深度图像上色以显示不同深度范围的渐变效果。

    :param depth_image: 输入的深度图像，应该是一个单通道（灰度）的2D数组
    :return: 彩色化的深度图像
    """
    colored_depth_image = depth_image.copy()  # 复制深度图像，避免修改原图
    # 确保深度图像是浮点型
    if colored_depth_image.dtype != np.float32:
        colored_depth_image = colored_depth_image.astype(np.float32)

    # 归一化深度图像到0-1范围
    depth_image_normalized = cv2.normalize(colored_depth_image, None, 0, 1, cv2.NORM_MINMAX)

    # 对归一化后的深度值进行伽马校正，以增强对比度
    depth_image_corrected = np.power(depth_image_normalized, 0.5)  # 0.5是伽马值，可调节
    
    # 转换为伪彩色图像
    color_depth_image = cv2.applyColorMap((depth_image_corrected * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return color_depth_image

