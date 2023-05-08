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
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox,DecodeBoxScript

class YOLO(object):
    # _defaults = {
    #     #--------------------------------------------------------------------------#
    #     #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #     #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #     #
    #     #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #     #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
    #     #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #     #--------------------------------------------------------------------------#
    #     # "model_path"        : 'model_data/yolov4_tiny_weights_coco.pth',
    #
    #      "model_path"        : 'model_data/yolov4_tiny_weights_lmk.pth',
    #     "classes_path"      : 'model_data/lmk_classes.txt',
    #     #---------------------------------------------------------------------#
    #     #   anchors_path代表先验框对应的txt文件，一般不修改。
    #     #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #     #---------------------------------------------------------------------#
    #     "anchors_path"      : 'model_data/yolo_anchors.txt',
    #     "anchors_mask"      : [[3,4,5], [1,2,3]],
    #     #-------------------------------#
    #     #   所使用的注意力机制的类型
    #     #   phi = 0为不使用注意力机制
    #     #   phi = 1为SE
    #     #   phi = 2为CBAM
    #     #   phi = 3为ECA
    #     #-------------------------------#
    #     "phi"               : 0,
    #     #---------------------------------------------------------------------#
    #     #   输入图片的大小，必须为32的倍数。
    #     #---------------------------------------------------------------------#
    #     # "input_shape"       : [416, 416],
    #
    #     # "input_shape":  [736, 864],
    #     "input_shape": [352, 416],
    #     #---------------------------------------------------------------------#
    #     #   只有得分大于置信度的预测框会被保留下来
    #     #---------------------------------------------------------------------#
    #     "confidence"        : 0.5,
    #     #---------------------------------------------------------------------#
    #     #   非极大抑制所用到的nms_iou大小
    #     #---------------------------------------------------------------------#
    #     "nms_iou"           : 0.3,
    #     #---------------------------------------------------------------------#
    #     #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #     #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #     #---------------------------------------------------------------------#
    #     "letterbox_image"   : False,
    #     #-------------------------------#
    #     #   是否使用Cuda
    #     #   没有GPU可以设置成False
    #     #-------------------------------#
    #     "cuda"              : False,
    # }
    #
    # @classmethod
    # def get_defaults(cls, n):
    #     if n in cls._defaults:
    #         return cls._defaults[n]
    #     else:
    #         return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, cfg):

        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = cfg.class_names, cfg.num_classes
        self.anchors, self.num_anchors      = cfg.anchors, cfg.num_anchors
        self.input_shape = cfg.input_shape
        self.anchors_mask = cfg.anchors_mask
        self.phi = cfg.phi
        self.model_path = cfg.model_path
        self.calc_device = cfg.calc_device
        self.confidence = cfg.confidence
        self.nms_iou = cfg.nms_iou
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
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.calc_device.type=="cuda":
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # #---------------------------------------------------#
    # #   检测图片
    # #---------------------------------------------------#
    # def detect_image(self, image):
    #     image_shape = np.array(np.shape(image)[0:2])
    #     #---------------------------------------------------------#
    #     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #     #---------------------------------------------------------#
    #     image       = cvtColor(image)
    #     #---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     #---------------------------------------------------------#
    #     image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
    #     #---------------------------------------------------------#
    #     #   添加上batch_size维度
    #     #---------------------------------------------------------#
    #     image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #         #---------------------------------------------------------#
    #         #   将图像输入网络当中进行预测！
    #         #---------------------------------------------------------#
    #         outputs = self.net(images)
    #         outputs = self.bbox_util.decode_box(outputs)
    #         #---------------------------------------------------------#
    #         #   将预测框进行堆叠，然后进行非极大抑制
    #         #---------------------------------------------------------#
    #         results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
    #                     image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
    #
    #         if results[0] is None:
    #             return image
    #
    #         top_label   = np.array(results[0][:, 6], dtype = 'int32')
    #         top_conf    = results[0][:, 4] * results[0][:, 5]
    #         top_boxes   = results[0][:, :4]
    #     #---------------------------------------------------------#
    #     #   设置字体与边框厚度
    #     #---------------------------------------------------------#
    #     font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #     thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
    #
    #     #---------------------------------------------------------#
    #     #   图像绘制
    #     #---------------------------------------------------------#
    #     for i, c in list(enumerate(top_label)):
    #         predicted_class = self.class_names[int(c)]
    #         box             = top_boxes[i]
    #         score           = top_conf[i]
    #
    #         top, left, bottom, right = box
    #
    #         top     = max(0, np.floor(top).astype('int32'))
    #         left    = max(0, np.floor(left).astype('int32'))
    #         bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
    #         right   = min(image.size[0], np.floor(right).astype('int32'))
    #
    #         label = '{} {:.2f}'.format(predicted_class, score)
    #         draw = ImageDraw.Draw(image)
    #         label_size = draw.textsize(label, font)
    #         label = label.encode('utf-8')
    #         print(label, top, left, bottom, right)
    #
    #         if top - label_size[1] >= 0:
    #             text_origin = np.array([left, top - label_size[1]])
    #         else:
    #             text_origin = np.array([left, top + 1])
    #
    #         for i in range(thickness):
    #             draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
    #         draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
    #         draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
    #         del draw
    #
    #     return image


    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#

    def visualize_bbox(self,img, bbox,score, class_name, color=(255, 0, 0), thickness=2):
        """Visualizes a single bounding box on the image"""
        # x_min, y_min, w, h = bbox
        y_min,x_min, y_max ,x_max=bbox
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        y_min_n =  y_min - int(1.3 * text_height)
        x_max_n = x_max + text_width
        cv2.rectangle(img, (int(x_min),int(y_min_n)), (int(x_max), int(y_min)), color=color, thickness=-1)
        cv2.putText(
            img,
            text= '{} {:.2f}'.format(class_name, score),
            org=(int(x_min), int(y_min) - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255),
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
            N = 100
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

            t = (t2-t1)/N
            print("predict time fps: ",t)

            if results[0] is None:
                return image_raw

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # # ---------------------------------------------------------#
        # #   设置字体与边框厚度
        # # ---------------------------------------------------------#
        # font = ImageFont.truetype(font='model_data/simhei.ttf',
        #                           size=np.floor(3e-2 * image.shape[1] + 0.5).astype('int32'))
        # thickness = int(max((image.shape[0] + image.shape[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        image = self.visualize(image_raw, top_label, top_conf, top_boxes)


        # for i, c in list(enumerate(top_label)):
        #     predicted_class = self.class_names[int(c)]
        #     box = top_boxes[i]
        #     score = top_conf[i]
        #
        #     top, left, bottom, right = box
        #
        #     top = max(0, np.floor(top).astype('int32'))
        #     left = max(0, np.floor(left).astype('int32'))
        #     bottom = min(image.shape[1], np.floor(bottom).astype('int32'))
        #     right = min(image.shape[0], np.floor(right).astype('int32'))
        #
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #     label = label.encode('utf-8')
        #     print(label, top, left, bottom, right)
        #
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])
        #
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
        #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
        #     draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        #     del draw

        return image





    # def get_FPS(self, image, test_interval):
    #     image_shape = np.array(np.shape(image)[0:2])
    #     #---------------------------------------------------------#
    #     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #     #---------------------------------------------------------#
    #     image       = cvtColor(image)
    #     #---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     #---------------------------------------------------------#
    #     image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
    #     #---------------------------------------------------------#
    #     #   添加上batch_size维度
    #     #---------------------------------------------------------#
    #     image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #         #---------------------------------------------------------#
    #         #   将图像输入网络当中进行预测！
    #         #---------------------------------------------------------#
    #         outputs = self.net(images)
    #         outputs = self.bbox_util.decode_box(outputs)
    #         #---------------------------------------------------------#
    #         #   将预测框进行堆叠，然后进行非极大抑制
    #         #---------------------------------------------------------#
    #         results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
    #                     image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
    #
    #     t1 = time.time()
    #     for _ in range(test_interval):
    #         with torch.no_grad():
    #             #---------------------------------------------------------#
    #             #   将图像输入网络当中进行预测！
    #             #---------------------------------------------------------#
    #             outputs = self.net(images)
    #             outputs = self.bbox_util.decode_box(outputs)
    #             #---------------------------------------------------------#
    #             #   将预测框进行堆叠，然后进行非极大抑制
    #             #---------------------------------------------------------#
    #             results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
    #                         image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
    #
    #     t2 = time.time()
    #     tact_time = (t2 - t1) / test_interval
    #     return tact_time
    #
    # def get_map_txt(self, image_id, image, class_names, map_out_path):
    #     f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
    #     image_shape = np.array(np.shape(image)[0:2])
    #     #---------------------------------------------------------#
    #     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #     #---------------------------------------------------------#
    #     image       = cvtColor(image)
    #     #---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     #---------------------------------------------------------#
    #     image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
    #     #---------------------------------------------------------#
    #     #   添加上batch_size维度
    #     #---------------------------------------------------------#
    #     image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #         #---------------------------------------------------------#
    #         #   将图像输入网络当中进行预测！
    #         #---------------------------------------------------------#
    #         outputs = self.net(images)
    #         outputs = self.bbox_util.decode_box(outputs)
    #         #---------------------------------------------------------#
    #         #   将预测框进行堆叠，然后进行非极大抑制
    #         #---------------------------------------------------------#
    #         results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
    #                     image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
    #
    #         if results[0] is None:
    #             return
    #
    #         top_label   = np.array(results[0][:, 6], dtype = 'int32')
    #         top_conf    = results[0][:, 4] * results[0][:, 5]
    #         top_boxes   = results[0][:, :4]
    #
    #     for i, c in list(enumerate(top_label)):
    #         predicted_class = self.class_names[int(c)]
    #         box             = top_boxes[i]
    #         score           = str(top_conf[i])
    #
    #         top, left, bottom, right = box
    #         if predicted_class not in class_names:
    #             continue
    #
    #         f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
    #
    #     f.close()
    #     return


class YOLOScript(nn.Module):

    def __init__(self,cfg):
        super(YOLOScript, self).__init__()
        # anchors_mask, num_classes, model_src
        self.anchors_mask = cfg.anchors_mask
        self.num_classes = cfg.num_classes
        # self.model_src = cfg.model_path
        self.nms_iou = cfg.nms_iou
        self.confidence = cfg.confidence
        # self.letterbox_image = False

        # self.num_classes = 1
        # self.anchors_mask =cfg.anchors_mask
        self.input_shape = cfg.input_shape
        # self.anchors = torch.Tensor([[10,14], [23, 27], [37, 58], [81, 82], [164, 164], [264, 164]])
        self.anchors= cfg.anchors
        self.net = YoloBodyS(self.anchors_mask, self.num_classes)
        self.net.load_state_dict(torch.load(cfg.model_path))
        self.net.eval()

        self.bbox_util = DecodeBoxScript(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                       self.anchors_mask)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def forward(self, images):
        image_shape = images.size()[2:]
        outputs = self.net(images)
        outputs = self.bbox_util.decode_box(outputs[0],outputs[1])
        # return torch.cat((outputs[0].squeeze(),outputs[1].squeeze()),dim=0)
        # ---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        # ---------------------------------------------------------#
        num_classes = torch.tensor(self.num_classes)
        input_shape = torch.tensor(self.input_shape)
        image_shape = torch.tensor(image_shape)
        confidence =  torch.tensor(self.confidence)
        nms_iou = torch.tensor(self.nms_iou)
        results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape,
                                                     image_shape, conf_thres=confidence,
                                                     nms_thres=nms_iou)


        return results

