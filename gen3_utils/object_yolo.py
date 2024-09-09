from ultralytics import YOLO
import cv2

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')  # 使用YOLOv8n模型

# 进行物体检测
def detect_object(image, object_name):
    results = model(image)
    boxes = results[0].boxes
    for box in boxes:
        if model.names[int(box.cls)] == object_name:
            return box.xyxy.cpu().numpy().astype(int)
    return None

# 从bounding box中求取抓取点
def get_grasp_point(depth_image, object_name):
    # 获取bounding box内的深度图像
    bbox = detect_object(depth_image, object_name)
    depth_roi = depth_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    if bbox is not None:
        # 在图像上绘制bounding box
        cv2.rectangle(depth_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow('RealSense Depth', cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

        # 计算深度图像的平均值作为抓取点深度
        depth_mean = np.mean(depth_roi[depth_roi != 0])

        # 计算抓取点的中心位置
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        return (center_x, center_y, depth_mean)
                

            
