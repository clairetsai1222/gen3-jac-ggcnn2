import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # 创建空间滤波器，平滑深度值
        self.spatial_filter = rs.spatial_filter()
        # 创建深度填充（Depth Fill）过滤器，填充深度图像中的0值
        self.hole_filling_filter = rs.hole_filling_filter()



    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        # 创建Align对象，用于对齐深度图像和彩色图像
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 应用深度填充过滤器
        filled_depth_frame = self.hole_filling_filter.process(depth_frame)
        # 应用空间滤波器
        filtered_depth_frame = self.spatial_filter.process(filled_depth_frame)
        
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, filtered_depth_image, color_image

    def release(self):
        self.pipeline.stop()