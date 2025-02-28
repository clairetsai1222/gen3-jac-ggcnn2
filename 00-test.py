import numpy as np
# 读取标定矩阵
TS_eye2base = []
with open("utils/calibrate/calibration.txt", 'r') as f:
    for line in f:
        # 去除行尾换行符并按空格分割字符串
        row = list(map(float, line.strip().split()))
        TS_eye2base.append(row)
TS_eye2base = np.array(TS_eye2base)

TS_eye2base2 = np.array([
                    [9.973575258016956768e-01, 4.452985062401254579e-02, -5.740259689354709066e-02, 3.361887444730505649e+03],
                    [-4.224477132857497319e-02, 9.982891183610328456e-01, 4.042543082452214331e-02, -4.528185238694334771e+02],
                    [5.910452624052021237e-02, -3.789364808717873845e-02, 9.975323184802228349e-01, 1.299203405789796761e+01],
                    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
                ])
print(TS_eye2base)
print(TS_eye2base2)