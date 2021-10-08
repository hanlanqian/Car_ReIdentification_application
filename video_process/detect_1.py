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
def detect_plate(img, device, model, modelc):
    # Initialize
    half = device.type != 'cpu'  # half precision only supported on CUDA

    im0s = img  # 这里应该是类似于底图的概念

    # 对底图进行一波操作，生成img
    # Padded resize
    img = letterbox(im0s, new_shape=opt.img_size)[0]
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
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    # Apply Classifier
    # pred是一个多维的tensor，plate_num是一个list的list
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
                # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                *xyxy, conf, cls = de  # 这里不知道cls的用处
                conf = conf.cpu().detach().numpy()
                conf = conf.astype(np.float32)

                for m, n in enumerate(xyxy):
                    a = n.cpu().detach().numpy().tolist()
                    lst.append(int(a))
                # xyxy是一个tensor，存储着车牌的坐标位置值

                for a, j in enumerate(lic_plat):
                    # if a ==0:
                    #     continue
                    lb += CHARS[int(j)]


        elif det is None:
            print("不存在车牌识别结果")
            continue

    return lb, lst, conf


# 处理图片
def process_image(i_image, o_image):
    # 创建存放输出图片的文件夹（若文件夹已存在，先删除）
    if os.path.exists(o_image):
        shutil.rmtree(o_image)  # delete output folder
    os.makedirs(o_image)  # make new output folder

    # 车牌识别模型的载入
    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = True
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
        print("load pretrained model successful!")
        modelc.to(device).eval()

    i_image = cv2.imread(i_image)
    img, classify, lst, vehicle_type, vehicle_conf = yolo.detect_image(Image.fromarray(i_image))
    img = np.array(img)
    # classify是一个bool变量，当检测出车辆时返回值为True，否则为False
    # lst是一个列表，返回值为检测出的车辆的坐标区域值

    # 创建一个Dataframe储存检测的结果
    # vehicle_area：车辆的坐标区域（对于原图像而言，list类型，四个元素值分别对应左上的xy坐标，右下的xy坐标）
    # vehicle_type：车辆的类型；vehicle_conf：车辆检测的置信度；plate：车牌
    # plate_area：车牌的坐标区域（对于截取的车辆图像而言，list类型，四个元素值分别对应左上的xy坐标，右下的xy坐标）
    # plate_conf：车牌识别的置信度；image_path：图片存储的路径
    df = pd.DataFrame(columns=['vehicle_area', 'vehicle_type',
                               'vehicle_conf', 'plate', 'plate_area', 'plate_conf', 'image_path'])

    count = 0  # 第cnt帧的检测出的图片数量
    expand_name = '.jpg'
    if classify:
        for i in range(len(lst)):
            count += 1

            top, left, bottom, right = lst[i][0], lst[i][1], lst[i][2], lst[i][3]
            vehicle_area = [top, left, bottom, right]
            # print(top, left, bottom, right)
            crop_img = img[int(top): int(bottom), int(left): int(right)]
            # crop_img的格式为<class 'numpy.ndarray'>

            # 车牌检测
            plate, plate_area, plate_conf = detect_plate(crop_img, device, model, modelc)

            image_name = str(count)
            image_path = os.path.join(o_image, image_name + expand_name)
            image_path = image_path.replace('\\', '/')
            cv2.imwrite(image_path, crop_img)
            df = df.append([{'vehicle_area': vehicle_area,
                             'vehicle_type': vehicle_type[i], 'vehicle_conf': vehicle_conf[i],
                             'plate': plate, 'plate_area': plate_area, 'plate_conf': plate_conf,
                             'image_path': image_path}, ], ignore_index=True)

    csv_path = os.path.join(o_image, 'output.csv')
    csv_path = csv_path.replace('\\', '/')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    df.to_csv(csv_path)
    print(df.head())


# 实现车辆目标检测，将车辆截取保存到文件夹
def object_detection(i_video, o_video, num):
    # 创建存放输出图片的文件夹（若文件夹已存在，先删除）
    if os.path.exists(o_video):
        shutil.rmtree(o_video)  # delete output folder
    os.makedirs(o_video)  # make new output folder

    # 准备截取视频
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # num_frame为视频的总帧数
    expand_name = '.jpg'
    if not cap.isOpened():
        print("Please check the path.")

    # 创建一个Dataframe储存检测的结果
    # cnt_num：截取的帧数；vehicle_area：车辆的坐标区域（对于原图像而言，list类型，四个元素值分别对应左上的xy坐标，右下的xy坐标）
    # vehicle_type：车辆的类型；vehicle_conf：车辆检测的置信度；image_path：图片存储的路径
    df = pd.DataFrame(columns=['cnt_num', 'vid_time', 'vehicle_area', 'vehicle_type',
                               'vehicle_conf', 'image_path'])

    # 计数变量等
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获得该视频的帧率
    car_type = ['car', 'bus', 'truck']
    cnt = 0  # 帧数

    while 1:
        ret, frame = cap.read()  # ret是一个bool变量，应该是判断是否成功截取当前帧；frame是截取的当前帧
        cnt += 1

        # old_lst = []  # 用于储存第n-1次的lst

        #  how many frame to cut
        if cnt % num == 0:
            print("正在处理第{}帧".format(cnt))

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
                    # print(top, left, bottom, right)
                    crop_img = frame[int(top): int(bottom), int(left): int(right)]
                    # crop_img的格式为<class 'numpy.ndarray'>

                    image_name = str(cnt) + str('_') + str(count)
                    image_path = os.path.join(o_video, image_name + expand_name)
                    image_path = image_path.replace('\\', '/')
                    cv2.imwrite(image_path, crop_img)
                    df = df.append([{'cnt_num': cnt, 'vid_time': vid_time, 'vehicle_area': vehicle_area,
                                     'vehicle_type': vehicle_type[i], 'vehicle_conf': vehicle_conf[i],
                                     'image_path': image_path}, ], ignore_index=True)
                    # cv2.imshow('crop_img', crop_img)
                    # cv2.waitKey(0)

        if not ret:
            break

    csv_path = os.path.join(o_video, 'output.csv')
    csv_path = csv_path.replace('\\', '/')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    df.to_csv(csv_path)
    print(df.head())


# 输入车辆检测的结果图片，进行车牌检测
def plate_recognition(o_video):
    # 车牌识别模型的载入
    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = True
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
        print("load pretrained model successful!")
        modelc.to(device).eval()

    # 存放车辆检测结果的csv文件的地址
    csv_path = os.path.join(o_video, 'output.csv')
    csv_path = csv_path.replace('\\', '/')
    df = pd.read_csv(csv_path, index_col=0)

    # plate：车牌
    # plate_area：车牌的坐标区域（对于截取的车辆图像而言，list类型，四个元素值分别对应左上的xy坐标，右下的xy坐标）
    # plate_conf：车牌识别的置信度
    col_name = df.columns.tolist()  # 将数据框的列名全部提取出来存放在列表里
    col_name.insert(5, 'plate')  # 在列索引为5的位置插入一列,列名为:plate，刚插入时不会有值，整列都是NaN
    col_name.insert(6, 'plate_area')
    col_name.insert(7, 'plate_conf')
    df = df.reindex(columns=col_name)  # DataFrame.reindex() 对原行/列索引重新构建索引值

    for i in range(df.shape[0]):
        image_path = df.iat[i, 8]
        img = cv2.imread(image_path)
        # 车牌检测
        plate, plate_area, plate_conf = detect_plate(img, device, model, modelc)
        df.iat[i, 5] = plate
        df.iat[i, 6] = str(plate_area)
        df.iat[i, 7] = plate_conf

    csv_path = os.path.join(o_video, 'output1.csv')
    csv_path = csv_path.replace('\\', '/')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    df.to_csv(csv_path)
    # print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # print(opt)

    t0 = time.time()
    yolo = YOLO()

    # 视频剪辑
    # clip = VideoFileClip("./inference/1-1-971.mp4").subclip(0, 300)
    # clip.write_videofile("./inference/1-1-971-cut.mp4")

    # process_image('./inference/002.jpg', './inference/output')
    object_detection("./inference/1-1-971-cut.mp4", './inference/output', 1)
    print('object detection is Done. (%.3fs)' % (time.time() - t0))
    plate_recognition('./inference/output')
    print('All is Done. (%.3fs)' % (time.time() - t0))
