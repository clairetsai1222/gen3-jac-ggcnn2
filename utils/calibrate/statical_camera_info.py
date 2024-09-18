import numpy as np

# 获取相机内参
def get_camera_intrinsics():
    depth_intrinsic_matrix = np.array([
        [427.52703857,   0.          , 426.38412476],
        [  0.          , 427.52703857, 237.69470215],
        [  0.          ,   0.        ,   1.        ]
    ])
    
    color_intrinsic_matrix = np.array([
        [910.4362793 ,   0.        , 647.40075684],
        [  0.        , 910.06066895, 364.79605103],
        [  0.        ,   0.        ,   1.        ]
    ])
    
    depth_scale = 999.999952502551
    
    return depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale

# 对齐深度和颜色传感器的变换矩阵
def align_depth_color_extrinsics():
    align_extrinsics_matrix = np.array([
        [-0.99435083, -0.08665283, -0.06130025, 12.55064651],
        [ 0.08201775, -0.9938447 ,  0.07447016,  5.07707018],
        [-0.06737597,  0.06902175,  0.99533737, 22.06476088],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    
    return align_extrinsics_matrix

# 获取相机外参
def get_camera_extrinsics():
    extrinsic_matrix = np.array([
        [  0.99362922,   0.05213893,   0.09991244, -11.99744475],
        [ -0.03018934,   0.97728664,  -0.2097604 ,  -4.39653127],
        [ -0.10857978,   0.20540777,   0.97263461,  26.78985042],
        [  0.        ,   0.        ,   0.        ,   1.        ]
    ])
    
    return extrinsic_matrix
