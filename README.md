# 建构步骤

1. 训练模型：(应该已经完成)使用train_ggcnn2.py脚本训练模型，训练好的模型会保存在./output/models文件夹下
2. 验证模型：导入模型并在realsense d435i上测试，将物体的抓取点绘制在深度图像和彩色图像上，验证模型的准确性

ggcnn2是能从深度图像输入中找出机械臂抓取点并输出的模型，以.pt格式存储，使用时需要先导入模型，然后将深度图像输入模型，输出抓取点坐标。

要求如下：
使用一个realsense d435i相机以每秒30帧的频率得到深度图像和彩色图像，并且将深度图像输入这个模型然后求取出抓取点，并绘制在深度图像和彩色图像上


x坐标：601.04479 y坐标：540.7194 z坐标：51.7276 rx（x轴旋转角度）：24.0 ry（y轴旋转角度）：26.8424

打印棋盘格的网站：
https://calib.io/pages/camera-calibration-pattern-generator

（记得打印时要选择“实际尺寸”，并且事先丈量好完整棋盘格的尺寸，并准备相应大小的打印纸，RL2实验室中有一个棋盘，上面贴有已经打印好的棋盘格，如果还在可以直接拿来用）


'''
pip install -r requirements.txt

# 安装gen3 python api
python3 -m pip install <whl relative fullpath name>.whl

# 安装pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装 quaternion
conda install -c conda-forge quaternion
'''


```
# yolo分级模型
yolov8n.pt (nano, smallest and fastest)
yolov8s.pt (small)
yolov8m.pt (medium)
yolov8l.pt (large)
yolov8x.pt (xlarge, largest and most accurate)
```

嘗試將resized_depth_image grasp 變成原本的depth_image