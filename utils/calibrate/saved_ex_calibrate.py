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

#设置标定板尺寸信息 单位（mm）
grid_size = 26.8
offset = 2.7

# 设置numpy的打印选项，以避免科学记数法
np.set_printoptions(suppress=True, precision=8)

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
        gripper_R = Ts[3:]
        gripper_T = Ts[:3]
        base_axis_R = np.array([gripper_R[0],gripper_R[1],gripper_R[2]]).reshape(-1)
        T_hand_in_base = np.array([gripper_T[1],gripper_T[2],gripper_T[0]]).reshape(-1)
        # gripper_axis_T, gripper_axis_R = xyz2zxy(gripper_T[0],gripper_T[1],gripper_T[2], gripper_R[0],gripper_R[1],gripper_R[2])
        R_hand_in_base= R.from_euler('xyz',np.array(base_axis_R),degrees=True).as_matrix()


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

    T_board_to_camera_scaled = [arr * 0.01 for arr in T_board_to_camera]

    # print("R_base_to_hand:\n",R_base_to_hand)
    # print("T_base_to_hand:\n",T_base_to_hand)
    # print("R_board_to_camera:\n", R_board_to_camera)
    # print("T_board_to_camera:\n", T_board_to_camera)

    R_camera_to_base,T_camera_to_base = cv2.calibrateHandEye(R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera_scaled,method=cv2.CALIB_HAND_EYE_TSAI)
    return R_camera_to_base,T_camera_to_base, R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera_scaled


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
        if not isinstance(board_in_camera, np.ndarray) or not isinstance(hand_in_base, np.ndarray):
            print("image!{} abandoned!".format(img_name))
            continue
        else:
            Ts_board_in_camera_all.append(board_in_camera)
            Ts_hand_in_base_all.append(hand_in_base)
            totle_num += 1
            
    print("totle calibration imagenum:",totle_num)

    R_camera_to_base,T_camera_to_base, R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera_scaled = calibrate_opencv(Ts_board_in_camera_all,Ts_hand_in_base_all)
    return R_camera_to_base,T_camera_to_base, R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera_scaled

def get_all_files(path, extension="*.png"):
    # 使用glob获取所有指定扩展名的文件
    files = glob.glob(os.path.join(path, extension))
    return files

def check_calibration(cam_to_base_RT, chess_to_cam_R, chess_to_cam_T, base_to_end_R, base_to_end_T):
    # 结果验证，原则上来说，每次结果相差较小
    for i in range(0,len(chess_to_cam_T)):
        RT_base_to_end=np.column_stack((base_to_end_R[i],base_to_end_T[i].reshape(3,1)))
        RT_base_to_end=np.row_stack((RT_base_to_end,np.array([0,0,0,1])))
        # print(RT_end_to_base)
        RT_chess_to_cam=np.column_stack((chess_to_cam_R[i],chess_to_cam_T[i].reshape(3,1)))
        RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
        # print(RT_chess_to_cam)
        RT_chess_to_end=RT_base_to_end@cam_to_base_RT@RT_chess_to_cam #棋盘格相对于机器人末端坐标系位姿，固定
        # RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
        print('第',i,'次')
        print(RT_chess_to_end)
        print('')


if __name__ == '__main__':
    #图片所在路径
    path = f'./ex_aruco_calibration/'
    R_camera_to_base,T_camera_to_base, R_base_to_hand,T_base_to_hand,R_board_to_camera,T_board_to_camera_scaled = calibrate(path)
    print(T_camera_to_base)
    Ts_camera_to_base = np.vstack((np.hstack((R_camera_to_base,T_camera_to_base)),np.array([0,0,0,1])))
    check_calibration(Ts_camera_to_base, R_board_to_camera, T_board_to_camera_scaled, R_base_to_hand, T_base_to_hand)
    #存储标定结果矩阵
    np.savetxt("calibration.txt", Ts_camera_to_base)
    print(Ts_camera_to_base)
