#-----------------------------------------------------------------------#
#   使用模型进行预测，并将预测结果以图片形式保存在路径：'data/{project}/rst_img'
#-----------------------------------------------------------------------#
import colorsys
import time
import cv2
import os
import numpy as np
import torch
import torch.nn as nn

from utils.config import Config

# from torchvision.ops import nms ##为啥C++调用本文件的时候，加该语句就报错了
from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block

attention_block = [se_block, cbam_block, eca_block]

#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(pretrained)
        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)
        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)
        # #raw
        # if 1 <= self.phi and self.phi <= 3:
        #     self.feat1_att      = attention_block[self.phi - 1](256)
        #     self.feat2_att      = attention_block[self.phi - 1](512)
        #     self.upsample_att   = attention_block[self.phi - 1](128)

    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        # if 1 <= self.phi and self.phi <= 3:
        #     feat1 = self.feat1_att(feat1)
        #     feat2 = self.feat2_att(feat2)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5)
        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        # if 1 <= self.phi and self.phi <= 3:
            # P5_Upsample = self.upsample_att(P5_Upsample)
        #debug
        # P4 = torch.cat((P5_Upsample,feat1),axis=1)
        P4 = torch.cat((P5_Upsample, feat1), dim=1)
        #debug
        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        return (out0,out1)
        # ##制作script的时候使用下面方式
        # A = out1[0:1, :, 0:11, 0:13]
        # B = out1[0:1, :, 11:, 13:]
        # return torch.cat((out0,A,B),dim=0)

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def _nms_t(detections_class,nms_thres):
    max_detections = []
    while detections_class.shape[0]:
        # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        max_detections.append(torch.unsqueeze(detections_class[0], 0))
        if len(detections_class) == 1:
            break
        ious = bbox_iou(max_detections[-1][:, :4], detections_class[1:, :4])[0]
        detections_class = detections_class[1:][ious < nms_thres]
    # if len(max_detections) == 0:
    #     return None
    max_detections = torch.cat(max_detections, dim=0)
    return max_detections

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[81,82],[135,169],[344,319]
        #   26x26的特征层对应的anchor是[10,14],[23,27],[37,58]
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            # -----------------------------------------------#
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # -----------------------------------------------#
            #   输入为416x416时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # -------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            # -------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]
            # scaled_anchors=[]
            # for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]:
            #     scaled_anchors.append((anchor_width / stride_w, anchor_height / stride_h))

            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            # -----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = torch.sigmoid(prediction[..., 4])
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,13,13
            # ----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            # ----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            # ----------------------------------------------------------#
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                # ------------------------------------------#
                # keep = nms(
                #     detections_class[:, :4],
                #     detections_class[:, 4] * detections_class[:, 5],
                #     nms_thres
                # )
                # max_detections = detections_class[keep]

                max_detections =  _nms_t(detections_class,nms_thres)
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


class YOLO(object):

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
        return img

    def detect_cv_image(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        image_shape = image.shape[0:2]
        # ---------------------------------------------------------#
        #   opencv 图像通道默认BGR
        # ---------------------------------------------------------#
        if image.ndim!=3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image_raw = image.copy()
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), cv2.INTER_CUBIC)
        image = np.expand_dims(np.transpose(image / 255.0, (2, 0, 1)),0)
        images = torch.from_numpy(image).type(torch.FloatTensor).to(self.calc_device)
        with torch.no_grad():
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
            t = N/(t2-t1)
            print("predict time fps: ",t)
            if results[0] is None:
                return image_raw
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        image = self.visualize(image_raw, top_label, top_conf, top_boxes)
        return image


def Predict():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    project = "lslm"  ##LMK

    cfg = Config(project)
    yolo = YOLO(cfg)
    img_type = 'jpg'
    img_dir = r'G:\DefectDataCenter\General_Data\{}\val'.format(project)
    img_path = [x for x in os.listdir(img_dir) if x.endswith(img_type)]
    t1 = time.time()
    for img_id, img_name in enumerate(img_path):
        img_id += 1
        img = os.path.join(img_dir, img_name)
        r_image = yolo.detect_cv_image(img)
        image_name = str(img_id) + "_img.bmp"
        dst_path = os.path.join(cfg.dst_dir, image_name)
        cv2.imwrite(dst_path, r_image)
    fps = (len(img_path) / (time.time() - t1))
    print("fps: ", fps)

def Predictt():
    print("fps: ")

if __name__ == "__main__":
    Predict()
    # elif mode == "video":
    #     capture = cv2.VideoCapture(video_path)
    #     if video_save_path!="":
    #         fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    #         size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #         out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    #
    #     ref, frame = capture.read()
    #     if not ref:
    #         raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    #
    #     fps = 0.0
    #     while(True):
    #         t1 = time.time()
    #         # 读取某一帧
    #         ref, frame = capture.read()
    #         if not ref:
    #             break
    #         # 格式转变，BGRtoRGB
    #         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #         # 转变成Image
    #         frame = Image.fromarray(np.uint8(frame))
    #         # 进行检测
    #         frame = np.array(yolo.detect_image(frame))
    #         # RGBtoBGR满足opencv显示格式
    #         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #
    #         fps  = ( fps + (1./(time.time()-t1)) ) / 2
    #         print("fps= %.2f"%(fps))
    #         frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    #         cv2.imshow("video",frame)
    #         c= cv2.waitKey(1) & 0xff
    #         if video_save_path!="":
    #             out.write(frame)
    #
    #         if c==27:
    #             capture.release()
    #             break
    #
    #     print("Video Detection Done!")
    #     capture.release()
    #     if video_save_path!="":
    #         print("Save processed video to the path :" + video_save_path)
    #         out.release()
    #     cv2.destroyAllWindows()
    #
    # elif mode == "fps":
    #     img = Image.open('img/street.jpg')
    #     tact_time = yolo.get_FPS(img, test_interval)
    #     print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    #
    # elif mode == "dir_predict":
    #     import os
    #     from tqdm import tqdm
    #
    #     img_names = os.listdir(dir_origin_path)
    #     for img_name in tqdm(img_names):
    #         if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             image_path  = os.path.join(dir_origin_path, img_name)
    #             image       = Image.open(image_path)
    #             r_image     = yolo.detect_image(image)
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             r_image.save(os.path.join(dir_save_path, img_name))
    #
    # else:
    #     raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
