import numpy as np
import torch
import os,configparser
import torch.backends.cudnn as cudnn

# ---------------------------------------------------#
#   获得先验框
# ---------------------------------------------------#

class Config:

    def __init__(self,project):
        self.yolo_type = 'yolo4'   #yolo7tiny  yolo4
        self.project = project
        self.root = 'G:/DefectDataCenter/ParseData/Detection/{}/'.format(project)
        self.ini_p = self.root + 'raw_data/Config/{}_{}_Para.ini'.format(project,self.yolo_type)
        self.anchors_path = self.root + 'raw_data/Config/{}_{}_anchors.txt'.format(project,self.yolo_type)

        self.dst_root = self.root + "Pytorch_Data/"
        self.pth_dst = self.dst_root+r'/{}_{}.pth'.format(self.yolo_type,project)
        self.pt_dst =self.dst_root+r'/{}_{}.pt'.format(self.yolo_type,project)
        self.onnx_dst =self.dst_root+r'/{}_{}.onnx'.format(self.yolo_type,project)
        self.predict_dst = '{}/predicted'.format(self.dst_root)
        os.makedirs(self.dst_root,exist_ok=True)
        os.makedirs(self.predict_dst, exist_ok=True)

        self.fepoch = 50
        self.all_epoch = 100
        self.lr = [1e-3,1e-4]
        self.batch_size = 8
        self.num_workers = 4
        self.split_ratio = 0.8
        self.confidence = 0.5
        self.nms_iou = 0.3
        self.parse_ini()
        self.parse_train_para()
        self.calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.yolo4tiny_anchors_mask = [[3, 4, 5], [0, 1, 2]]  ##?  为啥是self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        self.yolo7_anchors_mask = [[6,7,8],[3, 4, 5], [0, 1, 2]]  ##?  为啥是self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        self.num_classes = len(self.class_names)
        self.input_shape = [self.ImageSizeY, self.ImageSizeX]
        ##yolo4-tiny
        if self.yolo_type=='yolo4':
            from nets.yolo import YoloBody,YoloBodyS
            from nets.yolo_training import YOLOLoss
            self.phi = 0
            self.model = YoloBody(self.yolo4tiny_anchors_mask, len(self.class_names), phi=0)
            self.yolo_loss = YOLOLoss(self.anchors, len(self.class_names), [self.ImageSizeY, self.ImageSizeX], self.calc_device, self.yolo4tiny_anchors_mask).to(self.calc_device)
            self.anchors_mask = self.yolo4tiny_anchors_mask
            self.net =self.model
            # ##yolo7
        elif self.yolo_type=='yolo7':
            from nets.yolo7 import YoloBody
            from nets.yolo_training import YOLOLoss
            self.phi = 'l'
            self.model = YoloBody(self.yolo7_anchors_mask, len(self.class_names), phi='l').to(self.calc_device)
            self.yolo_loss = YOLOLoss(self.anchors, len(self.class_names),
                                      [self.ImageSizeY, self.ImageSizeX],  self.calc_device,self.yolo7_anchors_mask).to(self.calc_device)
            self.anchors_mask = self.yolo7_anchors_mask
            self.net = self.model


        elif self.yolo_type=='yolo7tiny':
            from nets.yolo7tiny import YoloBody
            from nets.yolo_training import YOLOLoss

            self.model = YoloBody(self.yolo7_anchors_mask, len(self.class_names)).to(self.calc_device)
            self.yolo_loss = YOLOLoss(self.anchors, len(self.class_names),
                                      [self.ImageSizeY, self.ImageSizeX],  self.calc_device,self.yolo7_anchors_mask).to(self.calc_device)
            self.anchors_mask = self.yolo7_anchors_mask
            self.net = self.model
        # self.pth_dst=''
        if os.path.exists(self.pth_dst):
            self.model.load_state_dict(torch.load(self.pth_dst, map_location='cuda'))
        self.model_train = self.model.train()
        if self.calc_device.type == "cuda":
            self.model_train = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()

    def parse_ini(self):
        cfg = configparser.ConfigParser()
        cfg.read(self.ini_p)
        self.ClassesPath = cfg.get(self.project,'ClassesPath')
        self.IconDir = cfg.get(self.project, 'IconDir')
        self.TrainDataInfoPath = cfg.get(self.project, 'TrainDataInfoPath')
        self.ValDataInfoPath = cfg.get(self.project, 'ValDataInfoPath')
        self.ImageSizeX = int(cfg.get(self.project, 'ImageSizeX'))
        self.ImageSizeY = int(cfg.get(self.project, 'ImageSizeY'))
        self.AugFreq = int(cfg.get(self.project, 'AugFreq'))
        self.TestDataRatio = float(cfg.get(self.project, 'TestDataRatio'))/100.0

    def parse_train_para(self):
        with open(self.TrainDataInfoPath) as f:
            self.train_lines = f.readlines()
        with open(self.ValDataInfoPath) as f:
            self.val_lines = f.readlines()
        with open(self.ClassesPath) as f:
            self.class_names =[x.strip() for x in f.readlines()]

        with open(self.anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.epoch_step      = len(self.train_lines) // self.batch_size
        self.epoch_step_val  = max(len(self.val_lines) // self.batch_size,1)


if __name__ == '__main__':
    project = 'COT'
    A = Config(project)