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
import time

from models.ggcnn2 import GGCNN2
from utils.dataset_processing import grasp, grocess_output, take_place_utils
from utils.calibrate import statical_camera_info
from utils.yolo import object_detection
from utils.d435i_depth_detect import realsense_depth

# 导入YOLO模型相关的库
from ultralytics import YOLO

# 加载相机内参
_, depth_scale, color_coefficients = statical_camera_info.get_camera_intrinsics()

# 创建RealSense管道
# Initialize Camera Intel Realsense
dc = realsense_depth.DepthCamera()




# 加载GGCNN模型
ggcnn_model = GGCNN2()
ggcnn_model.load_state_dict(torch.load('./ggcnn2_weights_jacquard/epoch_100_iou_97_statedict.pt'))

expecting_detected_object = input("Please input the label of the object you want to detect: ")


stop_flag = True
try:
    while stop_flag:

        ret, depth_image, color_image, depth_intrinsic, color_intrinsic = dc.get_frame()

        if not ret:
            continue

        # 调整深度图像和彩色图像尺寸
        resized_depth_image, resized_color_image = take_place_utils.resize_images(depth_image, color_image, (704, 1280))

        ori_resized_depth_image = resized_depth_image.copy()

        # 对齐深度图像和彩色图像(代码存在问题，待修复，直接使用intel api)
        # aligned_depth_image = align_depth_color.align_images(resized_depth_image, resized_color_image, depth_intrinsic, color_intrinsic, statical_camera_info.align_depth_color_extrinsics())

        # 进行目标检测
        detect_result = object_detection.ObjectDetection(detect_object=expecting_detected_object, color_image=resized_color_image, detpth_image=resized_depth_image, color_intrinsics=color_intrinsic)
        objects_dict, object_keys = detect_result.get_results()

        for key in filter(lambda k: expecting_detected_object in k, object_keys):
            xyxy = objects_dict[key]['xyxy']
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1, x2, y2 = x1 - 10, y1 - 10, x2 + 10, y2 + 10
            object_depth_image = resized_depth_image[y1:y2, x1:x2]
            object_color_image = resized_color_image[y1:y2, x1:x2]

            # 使用GGCNN模型预测抓取点
            with torch.no_grad():
                depthT = torch.from_numpy(object_depth_image.reshape(1, 1, y2 - y1, x2 - x1).astype(np.float32))
                grasp_imgs = ggcnn_model(depthT)

            q_img, ang_img, width_img = grocess_output.post_process_output(grasp_imgs[0], grasp_imgs[1], grasp_imgs[2], grasp_imgs[3])
            grasps = grasp.detect_grasps(q_img=q_img, ang_img=ang_img, width_img=width_img, no_grasps=1)

            for grasp_objects in grasps:
                # 获取当前时间，方便图像命名
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
                grasp_point = grasp_objects.as_gr.as_grasp
                rectangle_center = grasp_objects.as_gr.center
                crop_grasp_point = (min(rectangle_center[0], x2-x1), min(rectangle_center[1], y2-y1))
                grasp_point = (crop_grasp_point[0] + x1, crop_grasp_point[1] + y1)
                cv2.circle(object_color_image, crop_grasp_point, 5, (0, 255, 0), -1)
                cv2.imwrite(f'./grasp_output/arm_frame_grasp_image/object_color_image_{current_time}.png', object_color_image)

                print("2D camaera frame:", grasp_point, current_time)

                # 保存图像
                for img, suffix in zip((resized_color_image, resized_depth_image), ('color', 'depth')):
                    cv2.circle(img, grasp_point, 5, (0, 255, 0), -1)
                    if suffix == 'depth':
                        img = take_place_utils.colorize_depth_image(img)
                    cv2.imwrite(f'./grasp_output/arm_frame_grasp_image/{suffix}_image_{current_time}.png', img)

                # # 绘制伪色彩深度图
                # colored_depth_image = take_place_utils.colorize_depth_image(resized_depth_image)
                # # 显示上色后的深度图像
                # cv2.imshow('Colored Depth Image', colored_depth_image)
                # cv2.imwrite(f'./grasp_output/arm_frame_grasp_image/colored_depth_image_{current_time}.png', colored_depth_image)

                # 计算3D抓取点
                # grasp_point_3d = take_place_utils.calculate_3d_grasp_point(grasp_point, ori_resized_depth_image, depth_intrinsic_matrix, depth_scale)
                grasp_point_3d = dc.get_spot3D(grasp_point, ori_resized_depth_image, depth_intrinsic, depth_scale)
                if grasp_point_3d is None:
                    print("计算3D抓取点失败，深度值无效。")
                    exit()
                else:
                    print("3D camera frame:", grasp_point_3d)

                # 建构齐次坐标
                grasp_point_homogeneous = np.array([grasp_point_3d[0], grasp_point_3d[1], grasp_point_3d[2], 1])
                # 齐次坐标与外参矩阵相乘，以获取在机械臂坐标系下的坐标
                T_camera_to_robot = statical_camera_info.get_camera_extrinsics()
                grasp_point_robot = np.dot(T_camera_to_robot, grasp_point_homogeneous)
                # 机械臂坐标系下的抓取点
                grasp_point_robot_3d = grasp_point_robot[:3]

                print("3D arm base frame:", grasp_point_robot_3d)


                # 等待1秒
                time.sleep(2)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = False

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    dc.release()
    cv2.destroyAllWindows()
