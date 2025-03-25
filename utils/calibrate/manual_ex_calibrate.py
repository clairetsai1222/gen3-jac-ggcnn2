import numpy as np

def euler_to_rotation_matrix(yaw, pitch, roll, order='ZYX'):
    """
    将欧拉角转换为旋转矩阵（默认Z-Y-X顺序）
    :param yaw: 绕Z轴旋转角度（度）
    :param pitch: 绕Y轴旋转角度（度）
    :param roll: 绕X轴旋转角度（度）
    :param order: 旋转顺序，支持 'ZYX' 或 'XYZ'
    :return: 3x3旋转矩阵
    """
    # 将角度转换为弧度
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    
    # 单轴旋转矩阵
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 按顺序组合旋转矩阵
    if order == 'ZYX':
        R = Rz @ Ry @ Rx  # Z-Y-X顺序（外旋）
    elif order == 'XYZ':
        R = Rx @ Ry @ Rz  # X-Y-Z顺序（内旋）
    else:
        raise ValueError("Unsupported rotation order")
    return R

def build_transform_matrix(R, t):
    """
    构建齐次变换矩阵
    :param R: 3x3旋转矩阵
    :param t: 平移向量 [tx, ty, tz]
    :return: 4x4齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


# 示例参数
yaw = 0     # 绕Z轴偏航角（度）
pitch = 0   # 绕Y轴俯仰角（度）
roll = 0    # 绕X轴滚转角（度）
t = np.array([0.5, 1.0, 1.5])  # 平移向量

# 机械臂y坐标轴相对西的偏移角度，顺时针为正；
# 例如，如下 beta 约等于 70
#           y  ^ 北
#            \ |
#      beta ( \|
# 西 <--------------------- 东
#              |
#              | 南

beta = 10

# 输入
print("alpha为摄像头向右相对西的偏移角度(度), xyz为平移(mm);\n")
print("按顺序输入 alpha, x, y, z, 以逗号分隔:\n")

# 输出

# 计算旋转矩阵和标定矩阵
R = euler_to_rotation_matrix(yaw, pitch, roll, order='ZYX')
T = build_transform_matrix(R, t)

print("\n标定矩阵 T:\n", T)