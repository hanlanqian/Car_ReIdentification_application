import os
import shutil
import numpy as np
import random

dataset_path = r"C:\Users\hanlanqian\Desktop\car_reid\test\TEST"
output = r"C:\Users\hanlanqian\Desktop\car_reid\test_processed"
current_dirs = True
query_num = 20

os.makedirs(output, exist_ok=True)

for root, dirs, imgs in os.walk(dataset_path, topdown=True):
    if current_dirs:
        current_dirs = False
        continue
    pid = int(root.split('\\')[-1])
    count = 0
    for img in imgs:
        shutil.copy(os.path.join(root, img), os.path.join(output, str(pid) + f'_{count}_c1.jpg'))
        count += 1

img_num = len(os.listdir(output))
mask = np.full(img_num, fill_value=True)
for i in range(query_num):
    mask[random.choice(range(img_num))] = False

img_list = os.listdir(output)
os.makedirs(os.path.join(output, 'image_test'), exist_ok=True)
os.makedirs(os.path.join(output, 'image_query'), exist_ok=True)

for i, img in enumerate(img_list):
    if mask[i]:
        shutil.move(os.path.join(output, img), os.path.join(output, 'image_test', img))
    else:
        shutil.move(os.path.join(output, img), os.path.join(output, 'image_query', img))
