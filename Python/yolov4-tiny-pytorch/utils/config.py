import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
from nets.yolo import YoloBody
# ---------------------------------------------------#
#   获得先验框
# ---------------------------------------------------#


yolo_anchors_txt = [14,14,  37,37,  58,37,  81,82,  164,164,  264,164]

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    anchors = torch.tensor(anchors,dtype=torch.float32)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def read_annotation_path(train_at_p,val_at_p):
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_at_p) as f:
        train_lines = f.readlines()
    with open(val_at_p) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    return train_lines,val_lines

Detection_Dataset_Dir = 'G:/DefectDataCenter/ParseData/Detection/'
class Config:
    def __init__(self,project):
        ###训练参数设置
        self.project = project
        self.unique_para_initial(project)
        self.calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.calc_device =  "cuda"
        #当前路径由train.py决定
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.model_path = r'model_data/yolov4_tiny_weights_{}.pth'.format(project)
        self.pt_model_dst = r'model_data/yolov4_tiny_weights_{}.pt'.format(project)
        self.onnx_model_dst = r'model_data/yolov4_tiny_weights_{}.onnx'.format(project)

        self.anchors_mask = [[3, 4, 5], [0,1,2]]  ##?  为啥是self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        self.phi = 0                                #  phi = 0为不使用注意力机制,1为SE，2为CBAM，3为ECA
        self.fepoch = 25
        self.all_epoch = 50
        self.lr = [1e-3,1e-4]
        self.batch_size = 8
        self.num_workers = 4

        #   获得图片路径和标签
        self.classes_path = r'{}{}/raw_data/Classes.txt'.format(Detection_Dataset_Dir, project)
        self.train_at_p = r'{}{}/raw_data/ImgBoxes_train.txt'.format(Detection_Dataset_Dir,project)
        self.val_at_p =  r'{}{}/raw_data/ImgBoxes_val.txt'.format(Detection_Dataset_Dir,project)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.train_lines,self.val_lines = read_annotation_path(self.train_at_p, self.val_at_p)
        self.epoch_step      = len(self.train_lines) // self.batch_size
        self.epoch_step_val  = max(len(self.val_lines) // self.batch_size,1)
        self.model = YoloBody(self.anchors_mask, self.num_classes, phi=self.phi)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location='cuda'))
        self.model_train = self.model.train()
        if self.calc_device.type == "cuda":
            self.model_train = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()

        ####预测参数设置
        self.confidence = 0.5
        self.nms_iou = 0.3
        # self.dir_origin_path = "Detection_Dataset_Dir/{}/aug_data/img1".format(project)
        self.dst_dir = r'data/{}/rst_img'.format(project)
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)
    def unique_para_initial(self,project):
        if project=="LMK":
            self.input_shape = [352, 416]  # 对原始图片进行缩放，大小是32的倍数  HW
        elif project=="HW":
            self.input_shape = [224, 224]  # 对原始图片进行缩放，大小是32的倍数
        elif project == "DSW_random":
            self.input_shape = [288, 896]  # 对原始图片进行缩放，大小是32的倍数
        elif project == "VOC":
            self.input_shape = [512, 512]  # 对原始图片进行缩放，大小是32的倍数
        elif project == "lslm":
            self.input_shape = [512, 704]  # 对原始图片进行缩放，大小是32的倍数

