import numpy as np
import time
import cv2
import cv2.aruco as aruco
from statical_camera_info import *
import datetime 
import os



# 设置相机参数
depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale, color_coefficients = get_camera_intrinsics()


# 欧拉角转换为旋转矩阵
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,0,0],
                    [0,np.cos(theta[0]),-np.sin(theta[0])],
                    [0,np.sin(theta[0]),np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]),0,np.sin(theta[2])],
                    #[0,-1,0],
                    [0, 1, 0],
                    [-np.sin(theta[1]),0,np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                    [np.sin(theta[2]),np.cos(theta[2]),0],
                    [0, 0,1]])

    #return np.dot(np.dot(R_z,R_y),R_x)
    # combined rotation matrix
    R = np.dot(R_z, R_y.dot(R_x))
    return R




# 标定板特征点检测 input:image
# 将检测得到的特征点按照maker的id大小排序
def aruco_detect(frame, save_flag=False, path=None):
    origin_frame = frame.copy()
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("输入图像的通道数不正确，预计为3个通道。")
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    print("Aruco 字典成功获取。")
    parameters = aruco.DetectorParameters()
    
    # detectMarkers(...)
    #     detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
    #     mgPoints]]]]) -> corners, ids, rejectedImgPoints

    # lists of ids and the corners beloning to each id
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    ids = ids.flatten()
    
    #将标定板上的标志按id排序，保证与3D点一一对应
    corners_sort = np.zeros((len(corners)*4,2),np.float16)

    ids_sort_idx = np.argsort(ids, -1)#xiao->da


    for i, idx in enumerate(ids_sort_idx):
        corner = corners[idx][0]
        #print(corner[0][0], corner[0][1])
        corners_sort[i * 4, 0] = corner[0][0]
        corners_sort[i * 4, 1] = corner[0][1]
        corners_sort[i * 4+1, 0] = corner[1][0]
        corners_sort[i * 4+1, 1] = corner[1][1]
        corners_sort[i * 4+2, 0] = corner[2][0]
        corners_sort[i * 4+2, 1] = corner[2][1]
        corners_sort[i * 4+3, 0] = corner[3][0]
        corners_sort[i * 4+3, 1] = corner[3][1]

    # 绘制特征点
    for i,c in enumerate(corners_sort):
        # print(c)
        cv2.putText(frame, str(i+1), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    1, cv2.LINE_AA)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(0)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_flag:
        if key == ord('s'):  # 如果按下's'键
            # 获取当前系统时间并格式化
            
            img_name = f'{current_time}_aruco_image.png'  # 添加时间戳到文件名
            save_path = os.path.join(path,img_name)
            cv2.imwrite(save_path, origin_frame)  # 保存图像
            print(f"图像已保存至：{save_path}")
    
        cv2.waitKey(1)  # 防止窗口冻结
    else:
        cv2.waitKey(1)
    return corners_sort, origin_frame, current_time

# 生成标定板3D坐标(marker从左到右，从上到下的顺序，每个marker左上角-右上角-右下角-左下角的顺序生成)
def generate_3D_point(grid_size,distance,size = (7,5)):
    point_n = size[0]*size[1]*4
    point_3d = np.zeros((point_n,3),np.float16)
    #print(point_3d.shape)
    for i in range(point_n):
        lu_idx = int(np.floor(i/4))*4
        #print(lu_idx)
        if i%4==0:
            point_3d[i,0] = np.floor(i/4)%5*(grid_size+distance)
            point_3d[i,1] = np.floor(i/(4*size[1]))*(grid_size+distance)
        if i%4==1:
            point_3d[i,0] = point_3d[lu_idx,0]+grid_size
            point_3d[i,1] = point_3d[lu_idx,1]
        if i%4==2:
            point_3d[i, 0] = point_3d[lu_idx, 0] + grid_size
            point_3d[i, 1] = point_3d[lu_idx, 1] + grid_size
        if i%4==3:
            point_3d[i, 0] = point_3d[lu_idx, 0]
            point_3d[i, 1] = point_3d[lu_idx, 1] + grid_size

    return point_3d

#grid_size：每个marker矩形的长度
#distance：marker之间的距离长度
#size:标定板marker的行数，列数 eg：(7,5)
def camera_calibrate(grid_size,distance,frame,size = (7,5)):
    # w,h,c = frame.shape
    #生成标定板的3D point
    point_3d = generate_3D_point(grid_size, distance, size)
    #检测图片中标定板的2D point
    corners_2d, _, _ = aruco_detect(frame)

    #转为统一数据类型
    point_3d = [point_3d.astype('float32')]
    corners_2d = [corners_2d.astype('float32')]
    # 利用3D和2D对应点计算变换矩阵
    (success, rotation_vector, translation_v) = cv2.solvePnP(np.array(point_3d), np.array(corners_2d), color_intrinsic_matrix, color_coefficients,flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_v = cv2.Rodrigues(rotation_vector)[0]

    return rotation_v,translation_v

# 反投影误差
def calculate_rejection_error(corners_2d,point_3d,rotation_v,translation_v,camera_matrix,distortion_coefficient):

    total_error = 0
    for i in range(len(point_3d[0])):
        imgpoints2, _ = cv2.projectPoints(point_3d[0][i], np.array(rotation_v), np.array(translation_v), camera_matrix,
                                          distortion_coefficient)
        imgpoints2 = imgpoints2.flatten()
        # cv2.circle(frame, (int(imgpoints2[0]), int(imgpoints2[1])), 2, (255, 0, 0), 2)
        # cv2.circle(frame, (int(corners_2d[0][i, 0]), int(corners_2d[0][i, 1])), 2, (0, 0, 255), 2)
        error = cv2.norm(corners_2d[0][i, :], imgpoints2, cv2.NORM_L2)
        total_error += error
    #print("total error: ", total_error / len(point_3d))
    return total_error



if __name__ == '__main__':

    frame = cv2.imread("./ex_aruco_calibration_image/aruco_test.jpg", -1)
    if frame is None:
        print("图像未成功加载，请检查路径。")
    else:
        print("图像加载成功，通道数:", frame.shape)

    rvec,tvec = camera_calibrate(10,1,frame)

