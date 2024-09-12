import cv2
import pyrealsense2 as rs
import numpy as np


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
    resized_depth_image = cv2.resize(depth_image, target_size[::-1], interpolation=cv2.INTER_NEAREST)
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