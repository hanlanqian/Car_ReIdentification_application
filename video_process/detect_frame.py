import configparser

from yolo import YOLO
import argparse
import time
import math
from PIL import Image
import cv2
import os
import numpy as np
import pandas as pd
import shutil
from moviepy.editor import *

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *


# 判断第n次检测结果的矩形区域与第n-1次是否高度重合
def compare_area(lst, old_lst):
    # lst是list类型，它是第n次检测结果的其中一个检测目标
    # old_lst是list类型，它的元素还是list类型，储存着第n-1次检测结果的全部检测目标
    qualification = False  # 当qualification为True时，说明符合高度重复

    center1 = [int((lst[0] + lst[2]) / 2), int((lst[1] + lst[3]) / 2)]
    for i in range(len(old_lst)):
        a = old_lst[i]
        center2 = [int((a[0] + a[2]) / 2), int((a[1] + a[3]) / 2)]
        x_minus = abs(center1[0] - center2[0])
        y_minus = abs(center1[1] - center2[1])
        center_distance = int(math.sqrt(math.pow(x_minus, 2) + math.pow(y_minus, 2)))

        if 0 <= center_distance <= 3:
            qualification = True

    return qualification


# 车牌识别
def detect_plate(img, device, model, modelc, config: configparser.ConfigParser):
    # Initialize
    half = device.type != 'cpu'  # half precision only supported on CUDA

    im0s = img  # 这里应该是类似于底图的概念

    # 对底图进行一波操作，生成img
    # Padded resize
    img = letterbox(im0s, new_shape=config.getint("video_process", 'IMAGE_SIZE'))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # Convert to Tensor
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=config.getboolean("video_process", 'AUGMENT'))[0]
    # Apply NMS
    pred = non_max_suppression(pred, config.getfloat("video_process", 'CONF_THRES'),
                               config.getfloat("video_process", 'IOU_THRES'),
                               classes=config.getboolean("video_process", 'CLASSES'),
                               agnostic=config.getboolean("video_process", 'AGNOSTIC_NMS'))
    # Apply Classifier
    pred, plat_num = apply_classifier(pred, modelc, img, im0s)

    lst = []
    lb = ''
    conf = 0

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            im0 = im0s
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 车牌坐标重定位

            # Write results
            for de, lic_plat in zip(det, plat_num):
                *xyxy, conf, cls = de  # 这里不知道cls的用处
                conf = conf.cpu().detach().numpy()
                conf = conf.astype(np.float32)

                for m, n in enumerate(xyxy):
                    a = n.cpu().detach().numpy().tolist()
                    lst.append(int(a))

                for a, j in enumerate(lic_plat):
                    lb += CHARS[int(j)]


        elif det is None:
            # print("不存在车牌识别结果")
            continue

    return lb, lst, conf


# 处理视频，先进行车辆检测，再对车辆图片进行车牌识别
def process_video(i_video, conf: configparser.ConfigParser, signals: list, camera_id, num=20):
    yolo = YOLO()

    # 准备截取视频
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # num_frame为视频的总帧数
    signals[1].emit(f'即将开始处理视频，视频总帧数为{num_frame}')
    expand_name = '.jpg'
    if not cap.isOpened():
        print("Please check the path.")

    # 车牌识别模型的载入
    # Initialize
    device = torch_utils.select_device(conf.get('video_process', 'DEVICE'))
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(conf.get('video_process', 'YOLO_WEIGHT'), map_location=device)  # load FP32 model
    imgsz = check_img_size(conf.getint('video_process', 'IMAGE_SIZE'), s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = True
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(conf.get('video_process', 'LPR_WEIGHT'), map_location=torch.device('cpu')))
        print("load pretrained model successful!")
        modelc.to(device).eval()

    # 创建一个Dataframe储存检测的结果
    # cnt_num：截取的帧数；vehicle_area：车辆的坐标区域（对于原图像而言，list类型，四个元素值分别对应左上的xy坐标，右下的xy坐标）
    # vehicle_type：车辆的类型；vehicle_conf：车辆检测的置信度；plate：车牌
    # plate_area：车牌的坐标区域（对于截取的车辆图像而言，list类型，四个元素值分别对应左上的xy坐标，右下的xy坐标）
    # plate_conf：车牌识别的置信度；image_path：图片存储的路径
    df = pd.DataFrame(columns=['cnt_num', 'vid_time', 'vehicle_area', 'vehicle_type',
                               'vehicle_conf', 'plate', 'plate_area', 'plate_conf', 'image_path'])

    # 计数变量等
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获得该视频的帧率
    car_type = ['car', 'bus', 'truck']
    cnt = 0  # 帧数

    signals[1].emit(f"图像输出位置:{os.path.join(conf.get('video_process', 'OUTPUT'), 'image_test')}")
    gallery_folder = os.path.join(conf.get('video_process', 'OUTPUT'), 'image_test')
    if not os.path.exists(gallery_folder):
        os.makedirs(gallery_folder)

    while 1:
        ret, frame = cap.read()  # ret是一个bool变量，应该是判断是否成功截取当前帧；frame是截取的当前帧
        cnt += 1
        # old_lst = []  # 用于储存第n-1次的lst

        #  how many frame to cut
        if cnt % num == 0:

            # 发射视频处理进度信号
            # print(int(cnt / num_frame * 100))
            signals[0].emit(int(cnt / num_frame * 100))
            # 将帧数转为视频的时间
            number_of_seconds = math.floor(cnt / fps)
            vid_time = time.strftime("%M:%S", time.gmtime(number_of_seconds))

            frame1, classify, lst, vehicle_type, vehicle_conf = yolo.detect_image(Image.fromarray(frame))
            # classify是一个bool变量，当检测出车辆时返回值为True，否则为False
            # lst是一个列表，返回值为检测出的车辆的坐标区域值

            # 这一部分是判断视频中是否有静止不动的车辆，如果有，则剔除掉以不输出结果
            # 判断车辆静止不动的原则是：第n-1帧与第n帧检测的坐标区域有高度重合的部分（两个矩形的中心点距离满足一定条件）
            # 这里要注意各个list有可能共享内存的问题
            new_lst = []  # 用于储存“合格”的车辆坐标区域
            if cnt == num:  # 第一次检测的结果保留，不参与比较
                new_lst = old_lst = lst
            else:
                for i in range(len(lst)):
                    qualification = compare_area(lst[i], old_lst)
                    # qualification为True，说明old_lst的值“近似”包含lst[i]的值
                    if not qualification:
                        new_lst.append(lst[i])
                old_lst = lst

            count = 0  # 第cnt帧的检测出的图片数量
            if classify and (vehicle_type[0] in car_type):  # vehicle_type是一个list，里面包含一个str类型的值
                for i in range(len(new_lst)):
                    count += 1
                    top, left, bottom, right = new_lst[i][0], new_lst[i][1], new_lst[i][2], new_lst[i][3]
                    vehicle_area = [top, left, bottom, right]
                    signals[1].emit(str(vehicle_type[0] + f': {top}, {left}, {bottom}, {right}'))
                    crop_img = frame[int(top): int(bottom), int(left): int(right)]
                    # crop_img的格式为<class 'numpy.ndarray'>

                    # 车牌检测
                    plate, plate_area, plate_conf = detect_plate(crop_img, device, model, modelc, conf)

                    image_name = f"{cnt}_{count}_c{camera_id}"
                    image_path = os.path.join(gallery_folder, image_name + expand_name)
                    image_path = image_path.replace('\\', '/')
                    cv2.imwrite(image_path, crop_img)
                    df = df.append([{'cnt_num': cnt, 'vid_time': vid_time, 'vehicle_area': vehicle_area,
                                     'vehicle_type': vehicle_type[i], 'vehicle_conf': vehicle_conf[i],
                                     'plate': plate, 'plate_area': plate_area, 'plate_conf': plate_conf,
                                     'image_path': image_path}, ], ignore_index=True)

        if not ret:
            break

    csv_path = os.path.join(conf.get('video_process', 'OUTPUT'), 'output.csv')
    csv_path = csv_path.replace('\\', '/')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    df.to_csv(csv_path)
    signals[0].emit(100)
    signals[1].emit('视频处理完成！')
