import numpy as np
import os
import cv2
import glob
import math
import time
from marker_detect import camera_calibrate,aruco_detect,generate_3D_point,eulerAnglesToRotationMatrix
from quaternions import Quaternion as Quaternion
from scipy.spatial.transform import Rotation as R
import realsense_depth
import gen3_gripper_pose 

#设置标定板尺寸信息 单位（mm）
grid_size = 26.8
offset = 2.7

# 设置相机参数

# 设置相机参数
mtx = np.array([
        [916.28, 0, 650.7],
        [0, 916.088, 352.941],
        [0, 0, 1],
    ])

gp = gen3_gripper_pose.GripperPose()

# Initialize Camera Intel Realsense
dc = realsense_depth.DepthCamera()

def save_img_and_base_pose(path):
    # 等待相机数据
    while True:
        rec, depth_img, color_img = dc.get_frame()
        # cv2.imshow("color", color_img)
        _, img, current_time, saved_flag = aruco_detect(color_img, path)
        if saved_flag:
            saved_flag = False
            break

    if rec:
        cartesian_pose = gp.return_gripper_pose()
        # 将cartesian_pose的数值写入.txt文件
        file_name = f'{current_time}_gripper_position.txt'
        file_save_path = os.path.join(path, file_name)
        with open(file_save_path, 'w') as f:
            f.write(f"{cartesian_pose.x*1000} {cartesian_pose.y*1000} {cartesian_pose.z*1000} "
                    f"{cartesian_pose.theta_x} {cartesian_pose.theta_y} {cartesian_pose.theta_z}\n") # 平移单位mm，旋转为角度
    else:
        print("No image received!")
        # return None, None

        
def batch_save_img_and_base_pose(num, path=None):
    for i in range(num):
        save_img_and_base_pose(path)
    


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

    print("R_board_in_camera:\n",R_board_in_camera)
    print("T_board_in_camera:\n",T_board_in_camera)

    if R_board_in_camera is None:
        print("board not detected!")
        return False

    Ts_board_in_camera = np.zeros((4,4), np.float64)
    Ts_board_in_camera[:3,:3] = R_board_in_camera
    Ts_board_in_camera[:3,3] = np.array(T_board_in_camera).flatten()
    Ts_board_in_camera[3,3] = 1
    return Ts_board_in_camera

def get_Ts_hand_in_base(file):
    if not os.path.exists(file):
        print("file not exist!")
        return False
    with open(file,"r") as f:
        line = f.readline()
        Ts = line.split(" ")
        Ts = [float(i) for i in Ts]
        # R_hand_in_base= eulerAnglesToRotationMatrix(np.array(Ts[3:]))
        R_hand_in_base= R.from_euler('xyz',np.array(Ts[3:]),degrees=True).as_matrix()
        T_hand_in_base = Ts[:3]

        print("R_hand_in_base:\n",R_hand_in_base)
        print("T_hand_in_base:\n",T_hand_in_base)

        # R T拼接
        Ts_hand_in_base = np.zeros((4, 4), np.float64)
        Ts_hand_in_base[:3, :3] = R_hand_in_base
        Ts_hand_in_base[:3, 3] = np.array(T_hand_in_base).flatten()
        Ts_hand_in_base[3, 3] = 1
    return Ts_hand_in_base

# 根据计算和读取到的Ts_board_to_camera,Ts_hand_to_base，利用calibrateHandEye标定得到变换矩阵
def calibrate_opencv(Ts_board_to_camera,Ts_hand_to_base):
    n = len(Ts_hand_to_base)

    R_base_to_hand = []
    T_base_to_hand = []
    R_board_to_camera = []
    T_board_to_camera = []

    for i in range(n):
        Ts_base_to_hand = np.linalg.inv(Ts_hand_to_base[i])
        R_base_to_hand.append(np.array(Ts_base_to_hand[:3,:3]))
        T_base_to_hand.append(np.array(Ts_base_to_hand[:3,3]))
        R_board_to_camera.append(np.array(Ts_board_to_camera[i][:3,:3]))
        T_board_to_camera.append(np.array(Ts_board_to_camera[i][:3,3]))

    print("R_base_to_hand:\n",R_base_to_hand)
    print("T_base_to_hand:\n",T_base_to_hand)
    print("R_board_to_camera:\n", R_board_to_camera)
    print("T_board_to_camera:\n", T_board_to_camera)

    R_camera_to_base,T_camera_to_base = cv2.calibrateHandEye(R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera,method=cv2.CALIB_HAND_EYE_TSAI)
    return R_camera_to_base,T_camera_to_base


# 生成Ts_board_to_camera,Ts_hand_to_base，调用calibrate_opencv
def calibrate(path):
    imgs_name = glob.glob(os.path.join(path, "*.png"))
    files_name = glob.glob(os.path.join(path, "*.txt"))

    Ts_hand_in_base_all = []
    Ts_board_in_camera_all = []
    totle_num = 0

    for img_name,file_name in zip(imgs_name,files_name):
        board_in_camera = get_Ts_board_in_camera(img_name)
        hand_in_base = get_Ts_hand_in_base(file_name)
        # print(f"board_in_camera:\n{board_in_camera}\nhand_in_base:\n{hand_in_base}")
        if not isinstance(board_in_camera, np.ndarray) or not isinstance(hand_in_base, np.ndarray):
            print("image!{} abandoned!".format(img_name))
            continue
        else:
            Ts_board_in_camera_all.append(board_in_camera)
            Ts_hand_in_base_all.append(hand_in_base)
            totle_num += 1
            
    print("totle calibration imagenum:",totle_num)

    R_camera_to_base,T_camera_to_base = calibrate_opencv(Ts_board_in_camera_all,Ts_hand_in_base_all)
    return R_camera_to_base,T_camera_to_base

def get_all_files(path, extension="*.png"):
    # 使用glob获取所有指定扩展名的文件
    files = glob.glob(os.path.join(path, extension))
    return files


if __name__ == '__main__':
    #图片所在路径
    path = f'./debug_calibration/'
    num = input("saving numbers...\n")
    batch_save_img_and_base_pose(int(num), path=path)
    R_camera_to_base,T_camera_to_base = calibrate(path)
    print(T_camera_to_base)
    Ts_camera_to_base = np.vstack((np.hstack((R_camera_to_base,T_camera_to_base)),np.array([0,0,0,1])))
    #存储标定结果矩阵
    np.savetxt("calibration.txt", Ts_camera_to_base)
