import colorsys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
from nets.yolo import YoloBody,YoloBodyS

# from nets.yolo7 import YoloBody

from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox,DecodeBoxScript

class YOLO(object):

    #     #---------------------------------------------------------------------#
    #     #   输入图片的大小，必须为32的倍数。
    #     #---------------------------------------------------------------------#
    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, cfg):

        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = cfg.class_names, len(cfg.class_names)
        self.anchors, self.num_anchors      = cfg.anchors, len(cfg.anchors)
        self.input_shape = [cfg.ImageSizeY,cfg.ImageSizeX]
        self.anchors_mask = cfg.anchors_mask

        self.model_path = cfg.pth_dst
        self.calc_device = cfg.calc_device
        self.confidence = cfg.confidence
        self.nms_iou = cfg.nms_iou

        self.net = cfg.model

        self.letterbox_image = False
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.calc_device.type=="cuda":
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def visualize_bbox(self,img, bbox,score, class_name, color=(255, 0, 0), thickness=2):
        """Visualizes a single bounding box on the image"""
        # x_min, y_min, w, h = bbox
        # y_min,x_min, y_max ,x_max=bbox
        x_min, y_min, x_max, y_max = bbox

        img = np.ascontiguousarray(img)
        cls_index = self.class_names.index(class_name)

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=self.colors[cls_index], thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        y_min_n =  y_min - int(1.3 * text_height)
        x_max_n = x_max + text_width
        cv2.rectangle(img, (int(x_min),int(y_min_n)), (int(x_max), int(y_min)), color=(255,255,255), thickness=-1)

        cv2.putText(
            img,
            text= '{} {:.2f}'.format(class_name, score),
            org=(int(x_min), int(y_min) - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=self.colors[cls_index],
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize(self,img, top_label,top_conf, top_boxes):
        # img_d = np.transpose(img,(1,0,2))
        for i, c in list(enumerate(top_label)):
            class_name = self.class_names[int(c)]
            score = top_conf[i]
            bbox = top_boxes[i]
            img = self.visualize_bbox(img, bbox, score, class_name)
        # cv2.imshow("img",img)

        # plt.figure(figsize=(12, 12))
        # plt.axis('off')
        # plt.imshow(img)
        return img

    def detect_cv_image(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        image_shape = image.shape[0:2]
        # ---------------------------------------------------------#
        #   opencv 图像通道默认BGR
        # ---------------------------------------------------------#
        if image.ndim!=3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        # image = cvtColor(image)
        image_raw = image.copy()

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        # image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), cv2.INTER_CUBIC)
        image = np.expand_dims(np.transpose(image / 255.0, (2, 0, 1)),0)
        images = torch.from_numpy(image).type(torch.FloatTensor).to(self.calc_device)
        with torch.no_grad():
            # images = torch.from_numpy(image_data)
            # if self.cuda:
            #     images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            t1 = time.time()
            N = 1
            for i in range(N):
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                             nms_thres=self.nms_iou)
            t2 = time.time()

            fps = N/(t2-t1)
            print("predict time fps: ",fps)

            if results[0] is None:
                # print("results[0] is None")
                return image_raw," "

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        image = self.visualize(image_raw, top_label, top_conf, top_boxes)


        return image,results


class YOLOScript(nn.Module):

    def __init__(self,cfg):
        super(YOLOScript, self).__init__()
        # anchors_mask, num_classes, model_src
        self.anchors_mask = cfg.anchors_mask
        self.num_classes = cfg.num_classes
        self.nms_iou = cfg.nms_iou
        self.confidence = cfg.confidence
        self.input_shape = cfg.input_shape
        self.anchors= cfg.anchors
        self.net = cfg.net.to('cpu')
        # self.net = cfg.model
        self.net.load_state_dict(torch.load(cfg.pth_dst))
        self.net.eval()

        self.bbox_util = DecodeBoxScript(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                       self.anchors_mask)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def forward(self, images):
        image_shape = images.size()[2:]
        with torch.no_grad():
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            features = self.net(images)
        #可以正常输出
            layer_num = len(features)
            out_pt = []
            for i in range(layer_num):
                # output_1 = self.bbox_util.decode_box(features[i])
                # output_2 = self.bbox_util.decode_box(features[1])
                out_pt.append(self.bbox_util.decode_box(features[i]))
            # outputs = torch.cat((output_1,output_2),dim=1)
            outputs = torch.cat(out_pt, dim=1)

        results = self.bbox_util.non_max_suppression(outputs, self.num_classes,
                                                         self.input_shape,image_shape,
                                                         conf_thres=0.5,nms_thres=0.3)


        # outputs = self.net(images)
        # outputs = self.bbox_util.decode_box(outputs)
        # # return torch.cat((outputs[0].squeeze(),outputs[1].squeeze()),dim=0)
        # # ---------------------------------------------------------#
        # #   将预测框进行堆叠，然后进行非极大抑制
        # # ---------------------------------------------------------#
        # num_classes = torch.tensor(self.num_classes)
        # input_shape = torch.tensor(self.input_shape)
        # image_shape = torch.tensor(image_shape)
        # confidence =  torch.tensor(self.confidence)
        # nms_iou = torch.tensor(self.nms_iou)
        # results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape,
        #                                              image_shape, conf_thres=confidence,
        #                                              nms_thres=nms_iou)
        return results
# class YOLOScript(nn.Module):
#
#     def __init__(self,cfg):
#         super(YOLOScript, self).__init__()
#         # anchors_mask, num_classes, model_src
#         self.anchors_mask = cfg.anchors_mask
#         self.num_classes = cfg.num_classes
#         # self.model_src = cfg.model_path
#         self.nms_iou = cfg.nms_iou
#         self.confidence = cfg.confidence
#         # self.letterbox_image = False
#         # self.num_classes = 1
#         # self.anchors_mask =cfg.anchors_mask
#         self.input_shape = cfg.input_shape
#         # self.anchors = torch.Tensor([[10,14], [23, 27], [37, 58], [81, 82], [164, 164], [264, 164]])
#         self.anchors= cfg.anchors
#         self.net = YoloBodyS(self.anchors_mask, self.num_classes)
#         self.net.load_state_dict(torch.load(cfg.model_path))
#         self.net.eval()
#
#         self.bbox_util = DecodeBoxScript(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
#                                        self.anchors_mask)
#
#     # ---------------------------------------------------#
#     #   检测图片
#     # ---------------------------------------------------#
#     def forward(self, images):
#         image_shape = images.size()[2:]
#         outputs = self.net(images)
#
#         # return outputs
#
#         outputs = self.bbox_util.decode_box(outputs[0],outputs[1])
#
#
#         # return torch.cat((outputs[0].squeeze(),outputs[1].squeeze()),dim=0)
#         # ---------------------------------------------------------#
#         #   将预测框进行堆叠，然后进行非极大抑制
#         # ---------------------------------------------------------#
#         num_classes = torch.tensor(self.num_classes)
#         input_shape = torch.tensor(self.input_shape)
#         image_shape = torch.tensor(image_shape)
#         confidence =  torch.tensor(self.confidence)
#         nms_iou = torch.tensor(self.nms_iou)
#         results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape,
#                                                      image_shape, conf_thres=confidence,
#                                                      nms_thres=nms_iou)
#         return results

