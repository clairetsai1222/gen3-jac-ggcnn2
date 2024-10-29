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

# 获取相机外参
def get_camera_extrinsics():
    extrinsic_matrix = np.array([
        [ 0.88378649,  0.46149773, -0.07707974, -61.14851245],
        [-0.45080934,  0.79578246, -0.40435283,  36.11730996],
        [-0.12526921,  0.39210984,  0.91134927, -45.47868771],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
    ])
    return extrinsic_matrix

