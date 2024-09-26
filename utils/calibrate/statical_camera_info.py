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
    
    depth_scale = 0.1 # 999.999952502551

    color_coefficients = np.array([0. , 0. , 0. , 0. , 0.])
    
    return depth_intrinsic_matrix, color_intrinsic_matrix, depth_scale, color_coefficients

# 对齐深度和颜色传感器的变换矩阵
def align_depth_color_extrinsics():
    align_extrinsics_matrix = np.array([
        [-0.99435083, -0.08665283, -0.06130025, 12.55064651],
        [ 0.08201775, -0.9938447 ,  0.07447016,  5.07707018],
        [-0.06737597,  0.06902175,  0.99533737, 22.06476088],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    
    return align_extrinsics_matrix

# # 获取相机外参
# def get_camera_extrinsics():
#     extrinsic_matrix = np.array([
#         [ 0.28007536,  0.94247020, -0.18250399,  1.28192908],
#         [ -0.83660160,  0.14639142, -0.52788949, -18.15083488],
#         [ -0.47080310,  0.30053197,  0.82947271,  77.12988939],
#         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
#     ])
    
#     return extrinsic_matrix

# 获取相机外参
def get_camera_extrinsics():
    extrinsic_matrix = np.array([
        [ 0.985340944,  0.0345212451, -0.167067375, -9.37672469],
        [ 0.0101754741,  0.965675584,  0.259551781, -7.60324611],
        [ 0.170292936, -0.257446987,  0.951168421,  54.3210428],
        [ 0.000000000,  0.000000000,  0.000000000,  1.00000000]
    ])
    
    return extrinsic_matrix

