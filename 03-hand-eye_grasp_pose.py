'''
For learning purpose, I create 4 versions of the code for different progress completion.

Level 03: 
I have gotten specific object's 2D grasp point on depth image. 
Now I want to get the 3D grasp point on the real world. 
I need to use the camera intrinsic parameters and the grasp point on the 2D image to get the 3D grasp point.
Also, I will calibrate the camera with robot arm to get the extrinsic matrix. 
And utilize the extrinsic matrix to get the 3D grasp point in robot base frame.

execution summary:
1. use the camera intrinsic parameters to get the 3D grasp point on the real world.
2. calibrate the camera with robot arm to get the extrinsic matrix.
3. utilize the extrinsic matrix to get the 3D grasp point in robot base frame.
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
from utils.yolo import object_detection

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

# 加载GGCNN模型
ggcnn_model = GGCNN2()
ggcnn_model.load_state_dict(torch.load('./ggcnn2_weights_jacquard/epoch_100_iou_97_statedict.pt'))

expecting_detected_object = input("Please input the label of the object you want to detect: ")

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

        # 调整深度图像和彩色图像尺寸
        resized_depth_image, resized_color_image = take_place_utils.resize_images(depth_image, color_image, (704, 1280))

        # 对齐深度图像和彩色图像
        aligned_depth_image = align_depth_color.align_images(resized_depth_image, resized_color_image, depth_intrinsic_matrix, color_intrinsic_matrix, statical_camera_info.align_depth_color_extrinsics())

        # 进行目标检测
        detect_result = object_detection.ObjectDetection(detect_object=expecting_detected_object, color_image=resized_color_image, detpth_image=aligned_depth_image, color_intrinsics=color_intrinsic_matrix)
        objects_dict, object_keys = detect_result.get_results()

        for key in filter(lambda k: expecting_detected_object in k, object_keys):
            xyxy = objects_dict[key]['xyxy']
            x1, y1, x2, y2 = map(int, xyxy)

            object_depth_image = resized_depth_image[y1:y2, x1:x2]

            # 使用GGCNN模型预测抓取点
            with torch.no_grad():
                depthT = torch.from_numpy(object_depth_image.reshape(1, 1, y2 - y1, x2 - x1).astype(np.float32))
                grasp_imgs = ggcnn_model(depthT)

            q_img, ang_img, width_img = grocess_output.post_process_output(grasp_imgs[0], grasp_imgs[1], grasp_imgs[2], grasp_imgs[3])
            grasps = grasp.detect_grasps(q_img=q_img, ang_img=ang_img, width_img=width_img, no_grasps=1)

            for grasp_objects in grasps:
                grasp_point = grasp_objects.as_gr.as_grasp
                rectangle_center = grasp_objects.as_gr.center
                grasp_point = (rectangle_center[0] + x1, rectangle_center[1] + y1)

                # 画抓取点
                for img, title in zip((resized_color_image, aligned_depth_image), ('Color Image with Grasp Point', 'Depth Image with Grasp Point')):
                    cv2.circle(img, grasp_point, 5, (0, 255, 0), -1)
                    cv2.imshow(title, img)

                # 保存图像
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for img, suffix in zip((resized_color_image, aligned_depth_image), ('color', 'depth')):
                    cv2.imwrite(f'./grasp_output/yolo_grasp_image/{suffix}_image_{current_time}.png', img)

                # 计算3D抓取点
                grasp_point_3d = take_place_utils.calculate_3d_grasp_point(grasp_point, resized_depth_image, depth_intrinsic_matrix, depth_scale)
                print("3D Grasp Point:", grasp_point_3d)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = False

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
