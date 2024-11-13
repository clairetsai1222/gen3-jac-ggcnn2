import numpy as np

def euler_to_rotation_matrix(rx, ry, rz):
    """
    根据欧拉角 (rx, ry, rz) 生成旋转矩阵。
    :param rx: 绕x轴的旋转角度（弧度制）
    :param ry: 绕y轴的旋转角度（弧度制）
    :param rz: 绕z轴的旋转角度（弧度制）
    :return: 3x3的旋转矩阵
    """
    # 绕X轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # 绕Y轴的旋转矩阵
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # 绕Z轴的旋转矩阵
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵 (Rz * Ry * Rx)
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    
    return rotation_matrix

def create_homogeneous_matrix(rotation_matrix, translation_vector):
    """
    创建齐次转换矩阵（包含旋转和平移）。
    :param rotation_matrix: 3x3的旋转矩阵
    :param translation_vector: 3x1的平移向量
    :return: 4x4的齐次变换矩阵
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix

def transform_coordinates_and_angles(x2, y2, z2, rx2, ry2, rz2, rotation_matrix, translation_vector):
    """
    通过齐次转换矩阵将坐标 (x2, y2, z2) 和欧拉角 (rx2, ry2, rz2) 变换到新的坐标系。
    :param x2, y2, z2: 输入的3D坐标
    :param rx2, ry2, rz2: 输入的旋转角度（弧度制）
    :param rotation_matrix: 3x3的旋转矩阵
    :param translation_vector: 3x1的平移向量
    :return: 变换后的坐标 (x1, y1, z1) 和角度 (rx1, ry1, rz1)
    """
    # 转换位置
    coords_2_homogeneous = np.array([x2, y2, z2, 1])
    transformation_matrix = create_homogeneous_matrix(rotation_matrix, translation_vector)
    transformed_coords_homogeneous = np.dot(transformation_matrix, coords_2_homogeneous)
    transformed_coords = transformed_coords_homogeneous[:3]

    # 转换角度 (旋转矩阵和输入的角度旋转矩阵相乘)
    rotation_matrix_2 = euler_to_rotation_matrix(rx2, ry2, rz2)
    transformed_rotation_matrix = np.dot(rotation_matrix, rotation_matrix_2)
    
    # 提取新的欧拉角
    ry1 = np.arcsin(-transformed_rotation_matrix[2, 0])
    rx1 = np.arctan2(transformed_rotation_matrix[2, 1], transformed_rotation_matrix[2, 2])
    rz1 = np.arctan2(transformed_rotation_matrix[1, 0], transformed_rotation_matrix[0, 0])
    new_rotate = np.array([rx1, ry1, rz1])
    
    return transformed_coords, new_rotate

def xyz2zxy(x2, y2, z2, rx2, ry2, rz2):
    # 旋转矩阵（x1对应z2, y1对应x2, z1对应y2）
    rotation_matrix = np.array([
        [0, 0, 1],  # x1对应z2
        [1, 0, 0],  # y1对应x2
        [0, 1, 0]   # z1对应y2
    ])

    # 平移向量 (tx, ty, tz)
    translation_vector = np.array([0, 0, 0])

    # 计算变换后的坐标和角度
    new_coords, new_rotate = transform_coordinates_and_angles(x2, y2, z2, rx2, ry2, rz2, rotation_matrix, translation_vector)
    gripper_axis_T = np.array(new_coords)
    gripper_axis_R = np.array(new_rotate)

    return gripper_axis_T, gripper_axis_R



if __name__ == '__main__':
    # 示例用法
    x2, y2, z2 = 1, 2, 3
    rx2, ry2, rz2 = np.radians(10), np.radians(20), np.radians(30)

    new_coords, new_rotate = xyz2zxy(x2, y2, z2, rx2, ry2, rz2)
    
    print(f"Transformed coordinates: {new_coords}")
    print(f"Transformed angles: {new_rotate}")
