import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np


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


def _nms_t(detections_class, nms_thres):
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

            # FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            # LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角 
            #   batch_size,3,13,13
            # ----------------------------------------------------------#
            # grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            #     batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            # grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            #     batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).to(x.device)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).to(x.device)

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            # ----------------------------------------------------------#
            # anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            # anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

            anchor_w = torch.tensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = torch.tensor(scaled_anchors).index_select(1, LongTensor([1]))



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
        image_shape = np.array(image_shape)
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape[::-1], image_shape[::-1]], axis=-1)
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
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


class DecodeBoxScript(nn.Module):
    def __init__(self, anchors, num_classes, input_shape,
                 anchors_mask=torch.tensor([[6, 7, 8], [3, 4, 5], [0, 1, 2]], dtype=torch.float32)):
        super(DecodeBoxScript, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[81,82],[135,169],[344,319]
        #   26x26的特征层对应的anchor是[10,14],[23,27],[37,58]
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs_1):
        dpara_device = inputs_1.device
        outputs = []
        i = 0
        scaled_anchors_list = []
        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        # -----------------------------------------------#
        batch_size = inputs_1.size(0)
        input_height = inputs_1.size(2)
        input_width = inputs_1.size(3)
        # -----------------------------------------------#
        #   输入为416x416时
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------#
        stride_h = self.input_shape[0] / input_height
        stride_w = self.input_shape[1] / input_width
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        anchor_tmp = self.anchors[self.anchors_mask[i]]
        anchor_tmp_len = anchor_tmp.size()[0]
        for j in range(anchor_tmp_len):
            anchor_size = anchor_tmp[j]
            s_w = (anchor_size[0] / stride_w).unsqueeze(0).unsqueeze(1)
            s_h = (anchor_size[1] / stride_h).unsqueeze(0).unsqueeze(1)
            scaled_anchors_list.append(torch.cat((s_w, s_h), dim=1))
        scaled_anchors = torch.cat(scaled_anchors_list, dim=0)


        # for anchor_size in self.anchors[self.anchors_mask[i]]:
        #     s_w = (anchor_size[0] / stride_w).unsqueeze(0).unsqueeze(1)
        #     s_h = (anchor_size[1] / stride_h).unsqueeze(0).unsqueeze(1)
        #     scaled_anchors_list.append(torch.cat((s_w, s_h), dim=1))
        # scaled_anchors = torch.cat(scaled_anchors_list, dim=0)

        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        # -----------------------------------------------#
        prediction = inputs_1.view(batch_size, len(self.anchors_mask[i]),
                                   self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous().to(
            dpara_device)

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
        # ----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   batch_size,3,13,13
        # ----------------------------------------------------------#
        grid_x = torch.arange(input_width, dtype=torch.float32).repeat(input_height, 1).repeat(
            batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).to(dpara_device)
        grid_y = torch.arange(input_height, dtype=torch.float32).repeat(input_width, 1).t().repeat(
            batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).to(dpara_device)
        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size,3,13,13
        # ----------------------------------------------------------#
        # anchor_w = torch.index_select(scaled_anchors, dim=1, index=torch.tensor([0]))
        # anchor_h = torch.index_select(scaled_anchors, dim=1, index=torch.tensor([1]))
        # anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape).to(
        #     dpara_device)
        # anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape).to(
        #     dpara_device)
        anchor_w = scaled_anchors[:,0:1]
        anchor_h = scaled_anchors[:,1:2]
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape).to(
            dpara_device)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape).to(
            dpara_device)

        # ----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        # ----------------------------------------------------------#
        pred_boxes = torch.zeros(prediction[..., :4].shape).to(dpara_device)

        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        # ----------------------------------------------------------#
        #   将输出结果归一化成小数的形式
        # ----------------------------------------------------------#
        # _scale = torch.tensor([input_width, input_height, input_width, input_height], dtype=torch.float32).to(
        #     dpara_device)

        _scale = torch.stack((input_width, input_height, input_width, input_height),dim=0)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output

    def yolo_correct_boxes(self, box_xy, box_wh, image_shape):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = torch.zeros_like(box_xy)
        box_yx[:, 0] = box_xy[:, 1]
        box_yx[:, 1] = box_xy[:, 0]

        box_hw = torch.zeros_like(box_wh)
        box_hw[:, 0] = box_wh[:, 1]
        box_hw[:, 1] = box_wh[:, 0]

        # input_shape = torch.clone(input_shape)
        # image_shape = torch.clone(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        # boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
        #                        axis=-1)
        # np.concatenate([image_shape, image_shape], axis=-1)
        boxes = torch.cat(([box_mins[:, 0:1], box_mins[:, 1:2], box_maxes[:, 0:1], box_maxes[:, 1:2]]), dim=1)
        # boxes = boxes * torch.cat([torch.tensor(image_shape), torch.tensor(image_shape)], dim=0)

        b = torch.ones(2)
        b[0] ,b[1]= image_shape[0], image_shape[1]

        box_scale = b.repeat(2)
        boxes = boxes*box_scale
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, conf_thres,
                            nms_thres):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = torch.zeros(prediction.shape, dtype=torch.float32)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        image_pred = prediction[0]
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
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), dim=1)
        # ------------------------------------------#
        #   获得预测结果中包含的所有种类
        # ------------------------------------------#
        unique_labels = torch.unique(detections[:, -1])
        unique_labels_l = unique_labels.size()[0]
        # 统计每个图中图形
        output_c = torch.zeros(unique_labels_l, 7)

        max_detections = []
        # for unique_labels_id, c in enumerate(unique_labels):
        for k in range(unique_labels_l):
            c = unique_labels[k]
            unique_labels_id = k
            # ------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            # ------------------------------------------#
            detections_class = detections[detections[:, -1] == c]
            # # ------------------------------------------#
            # #   使用官方自带的非极大抑制会速度更快一些！
            # # ------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres)
            max_detections = detections_class[keep]
            # 按照存在物体的置信度排序
            if unique_labels_id == 0:
                output_c = max_detections
                # output[i].append(max_detections)
            else:
                output_c = torch.cat((output_c, max_detections), dim=0)
        box_xy = (output_c[:, 0:2] + output_c[:, 2:4]) / 2
        box_wh = output_c[:, 2:4] - output_c[:, 0:2]
        output_c[:, :4] = self.yolo_correct_boxes(box_xy, box_wh, image_shape)
        return output_c