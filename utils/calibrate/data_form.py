import numpy as np
import os
import cv2
import glob
import math
import time
from marker_detect import camera_calibrate,aruco_detect,generate_3D_point,eulerAnglesToRotationMatrix
from quaternions import Quaternion as Quaternion
from scipy.spatial.transform import Rotation as R
import statical_camera_info as scic
import realsense_depth
import gen3_gripper_pose 
import csv
import pandas as pd

#设置标定板尺寸信息 单位（mm）
grid_size = 26.8
offset = 2.7

# 设置相机参数

# 设置相机参数
mtx, depth_scale, coefficients = scic.get_camera_intrinsics()


# created by Leo Ma at ZJU, 2021.10.05
def get_Ts_board_in_camera(img_name):
    img = cv2.imread(img_name)
    if img is None:
        print("img read error!")
        return False
    w,h,c = img.shape
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
    # img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    R_board_in_camera,T_board_in_camera = camera_calibrate(grid_size,offset,img)

    # print("R_board_in_camera:\n",R_board_in_camera)
    # print("T_board_in_camera:\n",T_board_in_camera)

    if R_board_in_camera is None:
        print("board not detected!")
        return False

    R_board_in_camera = np.array(R.from_matrix(R_board_in_camera).as_euler('xyz', degrees=True))
    T_board_in_camera = np.array(T_board_in_camera * 0.01).flatten()
    board_in_camera = np.concatenate((T_board_in_camera.flatten(),R_board_in_camera.flatten()),axis=0)

    print("board_in_camera:\n",board_in_camera)

    return board_in_camera

def get_Ts_hand_in_base(file):
    if not os.path.exists(file):
        print("file not exist!")
        return False
    with open(file,"r") as f:
        line = f.readline()
        Ts = line.split(" ")
        Ts = [float(i) for i in Ts]
        # 将最后三个数值从角度转换为弧度
        # Ts[-3:] = np.radians(Ts[-3:])
        # R_hand_in_base= eulerAnglesToRotationMatrix(np.array(Ts[3:]))
        Ts = np.array(Ts).flatten()
       
        print("hand_in_base:\n",Ts)

    return Ts

# 生成Ts_board_to_camera,Ts_hand_to_base，调用calibrate_opencv
def calibrate(path):
    imgs_name = glob.glob(os.path.join(path, "*.png"))
    files_name = glob.glob(os.path.join(path, "*.txt"))

    Ts_hand_in_base_all = []
    Ts_board_in_camera_all = []
    data = []
    totle_num = 0

    for img_name,file_name in zip(imgs_name,files_name):
        board_in_camera = get_Ts_board_in_camera(img_name)
        hand_in_base = get_Ts_hand_in_base(file_name)
        if not isinstance(board_in_camera, np.ndarray) or not isinstance(hand_in_base, np.ndarray):
            print("image!{} abandoned!".format(img_name))
            continue
        else:
            # 使用列表推导式格式化数值，避免科学符号
            formatted_hand = [f"{num:4.6f}" for num in hand_in_base]  # 仍然保留6位小数，但最终可选择是否显示
            formatted_board = [f"{num:4.6f}" for num in board_in_camera]

            data.append(("hand",formatted_hand))
            data.append(("eye", formatted_board))
    csv_file_path = 'base_hand_to_eye_test_data.csv'

        # 写入 CSV 文件
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入数据行
        for row in data:
            writer.writerow(row)
    
    print(f"数据已成功写入 {csv_file_path}")

        # 读取原始CSV文件
    input_file = 'base_hand_to_eye_test_data.csv'
    output_file = 'base_hand_to_eye_test_data.csv'

    # 读取数据
    data = pd.read_csv(input_file, header=None, keep_default_na=False)

    # 创建一个列表来存储处理后的行
    processed_data = []

    # 遍历每一行，处理和转换数据
    for index, row in data.iterrows():
        key = row[0]  # 'hand' 或 'eye'
        values = row[1]  # 字符串格式的数组
        
        # 去除引号和方括号
        values = values.strip('"[]').replace(" ", ",").replace(",,,", ",").replace(",,", ",").replace("\n,", ",").replace(",,", ",")  # 替换空格为逗号
        values = values.replace("','", ",").replace("'", "").replace(",'", ",")  # 转换为数组
        # 创建新的格式
        processed_data.append(f"{key},{values}")

    # 写入新的CSV文件
    with open(output_file, 'w') as f:
        for line in processed_data:
            f.write(line + '\n')

    print(f"数据已成功转换并写入 {output_file}")

def get_all_files(path, extension="*.png"):
    # 使用glob获取所有指定扩展名的文件
    files = glob.glob(os.path.join(path, extension))

    return files


if __name__ == '__main__':
    #图片所在路径
    path = f'./ex_aruco_calibration/'
    calibrate(path)

