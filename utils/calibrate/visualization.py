
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取标定矩阵
TS_eye2base = []
with open("calibration.txt", 'r') as f:
    for line in f:
        row = list(map(float, line.strip().split()))
        TS_eye2base.append(row)
TS_eye2base = np.array(TS_eye2base)

if(1==0):
    TS_eye2base = np.array([
                    [0.893439, 0.401507, -0.201394, -60.01748],
                    [-0.443645, 0.858955, -0.255686, 0.907637],
                    [0.070328, 0.317787, 0.945550, 0.669926],
                    [0.000000, 0.000000, 0.000000, 1.000000]
                ])
    

R = TS_eye2base[:3, :3]  # 提取旋转矩阵
t = TS_eye2base[:3, 3]   # 提取平移向量

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制基座坐标系（红-X, 绿-Y, 蓝-Z）
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='Base X', length=50)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Base Y', length=50)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Base Z', length=50)

# 相机坐标系的原点在基座中的位置
origin = t

# 相机坐标系的轴方向（在基座中表示）
axes = np.array([[1,0,0], [0,1,0], [0,0,1]])  # 相机本地的X/Y/Z轴
rotated_axes = R @ axes  # 应用旋转变换
print(rotated_axes)
# 绘制相机坐标系
colors = ['r', 'g', 'b']
labels = ['Camera X', 'Camera Y', 'Camera Z']
for i in range(3):
    ax.quiver(
        origin[0], origin[1], origin[2],
        rotated_axes[0,i], rotated_axes[1,i], rotated_axes[2,i],
        color=colors[i], label=labels[i], length=50
    )

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()