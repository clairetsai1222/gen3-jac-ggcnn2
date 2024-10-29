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

        self.depth_frame = None
        self.color_frame = None

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        # 创建Align对象，用于对齐深度图像和彩色图像
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        if not self.depth_frame or not self.color_frame:
            return False, None, None
        self.depth_intrinsic = self.depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrinsic = self.color_frame.profile.as_video_stream_profile().intrinsics

        # 应用深度填充过滤器
        filled_depth_frame = self.hole_filling_filter.process(self.depth_frame)
        # 应用空间滤波器
        filtered_depth_frame = self.spatial_filter.process(filled_depth_frame)
        
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())


        return True, filtered_depth_image, color_image, self.depth_intrinsic, self.color_intrinsic

    def release(self):
        self.pipeline.stop()

    def get_spot3D(self, grasp_point_2d, depth_image, depth_intrinsic, depth_scale):
        """
        從depthimage中2D抓取點座標，计算3D抓取点。

        :param grasp_point_2d: 2D抓取点坐标 (x, y)
        :param depth_image: 深度图像
        :param depth_intrinsic_matrix: 深度相机的内参矩阵
        :param depth_scale: 深度图像的缩放因子
        :return: 3D抓取点 (x, y, z)
        """
        # 提取2D抓取点的坐标
        u, v = grasp_point_2d[1], grasp_point_2d[0]
        # 获取深度图像中对应点的深度值
        spot_depth = depth_image[int(u), int(v)] * depth_scale
        
        spot3D = rs.rs2_deproject_pixel_to_point(self.depth_intrinsic, [u, v], spot_depth)
        
        return np.array(spot3D)