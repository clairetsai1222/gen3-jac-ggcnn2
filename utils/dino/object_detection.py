

import cv2
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import torch
from typing import Tuple
from torchvision.ops import box_convert

import sys
sys.path.append(".../GroundingDINO")
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T

class ObjectDetection():
    def __init__(self, detect_object, color_image, detpth_image, color_intrinsics):
        self.intr = None
        self.color_image = color_image
        self.h, self.w, self._ = self.color_image.shape
        # print(self.h,self.w,self._)
        self.depth_image = detpth_image
        self.intr = color_intrinsics
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        self.color_image = self.image_format(self.color_image)
        # self.model = YOLO('yolov8n-seg.
        self.model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./GroundingDINO/weights/groundingdino_swint_ogc.pth")

        # bottle, cup, bowl, etc.
        self.desired_object = detect_object  # this will be able to be set by user

        self.frequency = 30.0
        self.objects_dict, self.object_keys = self.timer_callback()

    def timer_callback(self):

        # self.results = self.model.predict(self.color_image, conf=0.6, verbose=False, save_txt=False)
        boxes, logits, phrases = predict(
            model=self.model,
            image=self.color_image,
            caption=self.desired_object,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        # annotated_img = annotate(image_source=self.color_image, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imshow("Grounding Dino Inference", annotated_img)

        # 获取宽和高
        # 初始化 xywh 和 xyxy 变量
        xywh = None
        xyxy = None
        # 初始化物体存储字典来存储结果
        objects_dict = {}

        detect_num_flag = 0
        object_num_flag = 0
        ob_keys = []
        for b in boxes:
            # 检查类别名称是否匹配
            name = phrases[object_num_flag]
            if name == self.desired_object:
                # 获取 xywh 格式:边界框的中心点坐标 (x_center, y_center) 和边界框的宽度 width 和高度 height
                xywh = box_convert(boxes=b * torch.Tensor([self.w, self.h, self.w, self.h]), in_fmt="cxcywh", out_fmt="xywh").numpy()
                # 获取 xyxyn 格式:边界框的左上角坐标 (x_min, y_min) 和右下角坐标 (x_max, y_max)
                xyxy = box_convert(boxes=b * torch.Tensor([self.w, self.h, self.w, self.h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
                
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
    
    def image_format(self, image_source):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(Image.fromarray(image_source).convert("RGB"), None)
        return image_transformed

    def get_results(self):
        #print(self.objects_dict, self.object_keys)
        return self.objects_dict, self.object_keys
