import numpy as np

# 获取相机内参
def get_camera_intrinsics():
    intrinsic_matrix = np.array([
        [910.4362793 ,   0.        , 647.40075684],
        [  0.        , 910.06066895, 364.79605103],
        [  0.        ,   0.        ,   1.        ]
    ])
    depth_scale = 0.1 # 999.999952502551
    coefficients = np.array([0. , 0. , 0. , 0. , 0.])
    
    return intrinsic_matrix, depth_scale, coefficients


