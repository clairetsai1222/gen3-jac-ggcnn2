## 建构步骤

1. 训练模型：(应该已经完成)使用train_ggcnn2.py脚本训练模型，训练好的模型会保存在./output/models文件夹下
2. 验证模型：导入模型并在realsense d435i上测试，将物体的抓取点绘制在深度图像和彩色图像上，验证模型的准确性

ggcnn2是能从深度图像输入中找出机械臂抓取点并输出的模型，以.pt格式存储，使用时需要先导入模型，然后将深度图像输入模型，输出抓取点坐标。

要求如下：
使用一个realsense d435i相机以每秒30帧的频率得到深度图像和彩色图像，并且将深度图像输入这个模型然后求取出抓取点，并绘制在深度图像和彩色图像上


x坐标：601.04479 y坐标：540.7194 z坐标：51.7276 rx（x轴旋转角度）：24.0 ry（y轴旋转角度）：26.8424

打印棋盘格的网站：
https://calib.io/pages/camera-calibration-pattern-generator

（记得打印时要选择“实际尺寸”，并且事先丈量好完整棋盘格的尺寸，并准备相应大小的打印纸，RL2实验室中有一个棋盘，上面贴有已经打印好的棋盘格，如果还在可以直接拿来用）

## 标定步骤：
1. 连接机械臂和Realsense，安装棋盘格纸板
2. 运行标定程序
```
cd utils
cd calibrate
python auto_ex_calibrate.py
```
3. 图片量选择30张左右
4. 出现 “Available ids” 后按S保存（可自己建立 debug_calibration 文件夹）
5. 移动机械臂，用不同位姿重复第4步
6. 完成后，在 calibration.txt 中复制矩阵，填充到相关程序的对应位置。

## 配置步骤：

### 抓取部分：

#### 1. 安装 CUDA、Realsense SDK、连接机械臂等(略)
#### 2. 安装 pytorch
CUDA 11.8
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
CUDA 12.1
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
CUDA 12.4
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
#### 3. 安装其他依赖项
```
pip install -r requirements.txt
```
#### 4. 安装 Realsense d435i api
```
pip install pyrealsense2
```
#### 5. 安装 Gen3 python api
```
python3 -m pip install kortex_api-2.6.0.post3-py3-none-any.whl
```
#### 6. 安装 quaternion
可能需要增加镜像源：
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```
```
conda install -c conda-forge quaternion
```

### Grounding Dino部分:
#### 1. 进入 Grounding Dino 文件夹
```
cd GroundingDINO
```
#### 2. 初始化
```
pip install -e .
```
#### 3. 下载预训练模型
过大所以未上传至GitHub
```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## 运行：
```
cd ..
cd ..
python 04-grasp.py
```