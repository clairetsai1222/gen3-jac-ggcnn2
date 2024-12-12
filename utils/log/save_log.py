import os
import time
from PIL import Image

def Save(input_dict):
    # 获取当前时间作为文件夹名称
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件夹路径
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 创建以当前时间命名的子文件夹
    folder_path = os.path.join(log_dir, current_time)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # 遍历字典中的成员
    for index, (key, item) in enumerate(input_dict.items()):
        if isinstance(item, Image.Image):
            # 如果是图像对象，保存为图像文件
            image_path = os.path.join(folder_path, f'{key}.png')
            item.save(image_path)
        else:
            # 否则，保存为文本文件
            text_path = os.path.join(folder_path, f'{key}.txt')
            with open(text_path, 'w') as f:
                f.write(str(item))