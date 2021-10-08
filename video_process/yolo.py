# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4 import YoloBody
from utils.utils_1 import (DecodeBox, letterbox_image, non_max_suppression, non_max_suppression1,
                           yolo_correct_boxes)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class YOLO(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _defaults = {
        "model_path": 'video_process/model_data/yolo4_weights.pth',
        "anchors_path": 'video_process/model_data/yolo_anchors.txt',
        "classes_path": 'video_process/model_data/coco_classes.txt',
        "model_image_size": (416, 416, 3),
        "conf_thres": 0.65,  # 这里的置信度阈值是目标检测结果的阈值
        "iou_thres": 0.3,
        "cuda": True,
        "agnostic_nms": False,
        "max_det": 300,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   建立yolov4模型
        # ---------------------------------------------------#
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # ---------------------------------------------------#
        #   载入yolov4模型的权重
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.to(device)

        # ---------------------------------------------------#
        #   建立三个特征层解码用的工具
        # ---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # ---------------------------------------------------------#
        image = image.convert('RGB')
        classify = False

        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        images = [photo]

        lst = []
        class_lst = []
        score_lst = []
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.to(device)

            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, num_classes=len(self._get_class()),
                                                   conf_thres=self.conf_thres, nms_thres=self.iou_thres)

            # ---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                classify = True
            except:
                classify = False
                return image, classify, lst, class_lst, score_lst

            # ---------------------------------------------------------#
            #   对预测框进行得分筛选
            # ---------------------------------------------------------#
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.conf_thres
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            # -----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        Com = 0
        for i, c in enumerate(top_label):

            predicted_class = self.class_names[c]
            score = top_conf[i]
            # 计数部分
            if predicted_class == 'truck':  # 计数
                Com = Com + 1  # 计数
            # 计数部分
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
            lst1 = [top, left, bottom, right]
            lst.append(lst1)
            class_lst.append(predicted_class)
            score_lst.append(score)

            # 输出检测结果，包括类别，置信度，xyxy坐标
            label = '{} {:.2f}'.format(predicted_class, score)
            print(label, top, left, bottom, right)

        # 计数量
        # del draw JMW注释的 删除图片 预测图片用的如果用“video模式就需要del draw 一帧一帧看否则可能就是静止的”
        return image, classify, lst, class_lst, score_lst

    def get_FPS(self, image, test_interval):
        # 调整图片使其符合输入要求
        image_shape = np.array(np.shape(image)[0:2])

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.cuda:
                images = images.to(device)
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, num_classes=len(self._get_class()),
                                                   conf_thres=self.conf_thres, nms_thres=self.iou_thres)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                top_label = np.array(batch_detections[top_index, -1], np.int32)
                top_bboxes = np.array(batch_detections[top_index, :4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                    top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                               np.array([self.model_image_size[0], self.model_image_size[1]]),
                                               image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

            except:
                pass

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, num_classes=len(self._get_class()),
                                                       conf_thres=self.conf_thres, nms_thres=self.iou_thres)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index, -1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index, :4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3],
                                                                                                    -1)

                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                                   image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

                except:
                    pass

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
