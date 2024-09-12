#!/usr/bin/env python

import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np

class ObjectDetection():
    def __init__(self, detect_object, color_image, detpth_image, color_intrinsics):
        self.intr = None
        self.color_image = color_image
        self.depth_image = detpth_image
        self.intr = color_intrinsics

        self.model = YOLO('yolov8n-seg.pt')

        # bottle, cup, bowl, etc.
        self.desired_object = detect_object  # this will be able to be set by user

        self.frequency = 30.0
        self.objects_dict, self.object_keys = self.timer_callback()

    def timer_callback(self):
        self.results = self.model.predict(self.color_image, conf=0.6, verbose=False, save_txt=False)
        annotated_img = self.results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_img)

        # 初始化 xywh 和 xyxy 变量
        xywh = None
        xyxy = None
        # 初始化物体存储字典来存储结果
        objects_dict = {}

        detect_num_flag = 0
        object_num_flag = 0
        ob_keys = []
        for r in self.results:
            for c in r.boxes.cls:
                # 检查类别名称是否匹配
                if self.model.names[int(c)] == self.desired_object:
                    # 获取 xywh 格式:边界框的中心点坐标 (x_center, y_center) 和边界框的宽度 width 和高度 height
                    xywh = r.boxes.xywh.tolist()[object_num_flag]
                    # 获取 xyxyn 格式:边界框的左上角坐标 (x_min, y_min) 和右下角坐标 (x_max, y_max)
                    xyxy = r.boxes.xyxy.tolist()[object_num_flag]
                    
                    # 结合对象名称和数字为一个字符串
                    key = f"{self.desired_object}_{detect_num_flag}"
                    ob_keys.append(key)
                    # 存储到字典中
                    objects_dict[key] = {
                        'xywh': xywh,
                        'xyxy': xyxy
                    }
                    detect_num_flag += 1
                object_num_flag += 1
            object_num_flag = 0

        return objects_dict, ob_keys

    def get_results(self):
        return self.objects_dict, self.object_keys