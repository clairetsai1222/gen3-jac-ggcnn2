#!/usr/bin/env python

import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np

class ObjectDetection:
    def __init__(self):
        # realsense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(config)

        # passthrough filter publisher
        self.passthrough_vals = {
            'left': None,
            'right': None,
            'top': None,
            'bottom': None,
            'center_depth': None
        }

        self.intr = None
        self.color_image = None
        self.depth_image = None

        self.model = YOLO('yolov8n-seg.pt')

        # bottle, cup, bowl, etc.
        self.desired_object = "bottle"  # this will be able to be set by user

        self.frequency = 30.0
        self.timer_callback()

    def timer_callback(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            self.color_image = np.asanyarray(color_frame.get_data())
            self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
            self.depth_image = np.asanyarray(depth_frame.get_data())

            self.intr = color_frame.profile.as_video_stream_profile().intrinsics

            self.results = self.model.predict(self.color_image, conf=0.7, verbose=False, save_txt=False)
            annotated_img = self.results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_img)

            for r in self.results:
                for c in r.boxes.cls:
                    if (self.model.names[int(c)]) == self.desired_object:
                        xywh = r.boxes.xywh.tolist()[0]
                        xcenter, ycenter, width, height = [int(i) for i in xywh]

                        xleft = int(xcenter - width/2)
                        xright = int(xcenter + width/2)
                        ytop = int(ycenter + height/2)
                        ybottom = int(ycenter - height/2)

                        depth = self.depth_image[int(ycenter)][int(xcenter)]

                        if depth == 0:
                            continue

                        left = rs.rs2_deproject_pixel_to_point(self.intr, [xleft, ycenter], depth)
                        right = rs.rs2_deproject_pixel_to_point(self.intr, [xright, ycenter], depth)
                        top = rs.rs2_deproject_pixel_to_point(self.intr, [xcenter, ytop], depth)
                        bottom = rs.rs2_deproject_pixel_to_point(self.intr, [xcenter, ybottom], depth)

                        self.passthrough_vals['left'] = [i/1000. for i in left]
                        self.passthrough_vals['right'] = [i/1000. for i in right]
                        self.passthrough_vals['top'] = [i/1000. for i in top]
                        self.passthrough_vals['bottom'] = [i/1000. for i in bottom]
                        self.passthrough_vals['center_depth'] = depth/1000.

                        print(self.passthrough_vals)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

def main():
    obj_det = ObjectDetection()

if __name__ == '__main__': 
    try: 
        main() 
    except KeyboardInterrupt: 
        pass
