import numpy as np
import os
import cv2
import glob
import math
from marker_detect import camera_calibrate,aruco_detect,generate_3D_point,eulerAnglesToRotationMatrix
from quaternions import Quaternion as Quaternion
from scipy.spatial.transform import Rotation as R

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

dist = np.array([0, 0, 0, 0, 0])


# created by Leo Ma at ZJU, 2021.10.05
def get_Ts_board_in_camera(img_name):
    img = cv2.imread(img_name)
    if img is None:
        print("img read error!")
        return -1
    w,h,c = img.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    R_board_in_camera,T_board_in_camera = camera_calibrate(grid_size,offset,img)

    Ts_board_in_camera = np.zeros((4,4),np.float)
    Ts_board_in_camera[:3,:3] = R_board_in_camera
    Ts_board_in_camera[:3,3] = np.array(T_board_in_camera).flatten()
    Ts_board_in_camera[3,3] = 1
    return Ts_board_in_camera


def get_Ts_hand_in_base(file):
    if not os.path.exists(file):
        print("file not exist!")
        return -1
    with open(file,"r") as f:
        line = f.readline()
        Ts = line.split(" ")
        Ts = [float(i) for i in Ts]
        #R_hand_in_base= eulerAnglesToRotationMatrix(np.array(Ts[3:]))
        R_hand_in_base= R.from_euler('xyz',np.array(Ts[3:]),degrees=True).as_matrix()

        T_hand_in_base = Ts[:3]
        # R T拼接
        Ts_hand_in_base = np.zeros((4, 4), np.float)
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

    # R_hand_to_base = []
    # T_hand_to_base = []
    # R_camera_to_board = []
    # T_camera_to_board = []


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

    # R_cam2gripper,T_cam2gripper = calibrateHandEye(R_gripper2base,T_gripper2base,R_target2camera,T_target2camera):
    #     R_gripper2base : R_base_to_hand 
    #     T_gripper2base : T_base_to_hand 
    #     R_target2camera : R_board_to_camera
    #     T_target2camera : T_board_to_camera
    #     R_cam2gripper : R_camera_to_base
    #     T_cam2gripper : T_camera_to_base

    R_camera_to_base,T_camera_to_base = cv2.calibrateHandEye(R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera,method=cv2.CALIB_HAND_EYE_TSAI)

    #验证
    # Ts_camera_to_base = np.zeros((4, 4), np.float)
    # Ts_camera_to_base[:3, :3] = R_camera_to_base
    # Ts_camera_to_base[:3, 3] = np.array(T_camera_to_base).flatten()
    # Ts_camera_to_base[3, 3] = 1
    # print("-----------------验证-------------------")
    # print(np.dot(Ts_board_to_camera[0],np.dot(Ts_camera_to_base,np.linalg.inv(Ts_hand_to_base[0]))))
    # print(np.dot(Ts_board_to_camera[1], np.dot(Ts_camera_to_base, np.linalg.inv(Ts_hand_to_base[1]))))
    return R_camera_to_base,T_camera_to_base


# 生成Ts_board_to_camera,Ts_hand_to_base，调用calibrate_opencv
def calibrate(path):
    imgs_name = get_all_files(path, "*.png")
    files_name = get_all_files(path, "*.txt")

    Ts_hand_in_base_all = []
    Ts_board_in_camera_all = []

    for img_name,file_name in zip(imgs_name,files_name):
        Ts_board_in_camera_all.append(get_Ts_board_in_camera(os.path.join(path,img_name)))
        Ts_hand_in_base_all.append(get_Ts_hand_in_base(os.path.join(path,file_name)))

    R_camera_to_base,T_camera_to_base = calibrate_opencv(Ts_board_in_camera_all,Ts_hand_in_base_all)
    return R_camera_to_base,T_camera_to_base

def get_all_files(path, extension="*.png"):
    # 使用glob获取所有指定扩展名的文件
    image_files = glob.glob(os.path.join(path, extension))
    return image_files


if __name__ == '__main__':
    #图片所在路径
    path = f'./ex_aruco_calibration_images'
    R_camera_to_base,T_camera_to_base = calibrate(path)
    print(T_camera_to_base)
    Ts_camera_to_base = np.vstack((np.hstack((R_camera_to_base,T_camera_to_base)),np.array([0,0,0,1])))
    #存储标定结果矩阵
    np.savetxt("calibration.txt", Ts_camera_to_base)