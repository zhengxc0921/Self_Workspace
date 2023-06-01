import os
import torch
import random
import pandas as pd
import time
import cv2
import numpy as np
import shutil
from image_classifier.dataset import WaferImage
from torch.utils.data import DataLoader
from net.resnet import ResNet18

class Argu:
    def __init__(self,calc_device,use_feature=False,use_mask=False):
        self.use_feature = use_feature
        self.use_mask = use_mask

        # 配置参数：
        self.USE_MIL_Data = True
        if calc_device=='':
            self.calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.calc_device = calc_device
        self.input_size = (224, 224)
        self.damage_fold = 'FZ'
        self.split_ratio=0.8
        self.batchsz = 64
        self.lr =  0.1/10 * 1e-3

        self.src_root = 'G:\DefectDataCenter\ParseData\Classifier'
        self.dst_root = r'G:\DefectDataCenter\DeepLearningDataSet\Output'

        self.src_dir = os.path.join(self.src_root, self.damage_fold)
        self.model_name =  str(self.calc_device) + '_'+self.damage_fold
        self.model_dst_dir = r'../result/'+self.damage_fold
        self.best_model_name = self.model_name + '_best' + '.pkl'
        self.onnx_m_name = self.model_name  + '.onnx'
        self.pt_m_name = self.model_name + '.pt'
        self.best_model_path = os.path.join(self.model_dst_dir, self.best_model_name)
        self.onnx_m_dst = os.path.join(self.model_dst_dir,self.onnx_m_name)
        self.pt_m_dst = os.path.join(self.model_dst_dir,self.pt_m_name)


        self.label2class  = {}
        self.name2label = {}
        self.name2path = {}
        self.name2path_train = {}
        self.name2path_val = {}
        self.class_num_ratio = []  # 长尾问题，比例调整
        #解析mil数据集
        if self.USE_MIL_Data:
            self.work_space = r'G:\DefectDataCenter\ParseData\Classifier\SXX_GrayWave\DataSet'
            self.mil_data_initial()
        ##使用常规数据集
        else:
            self.pytorch_data_initial()

        self.folder_initial()
        self.model = ResNet18(num_classes=len(self.name2label)).to(self.calc_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def parse_class_definitions(self):
        cls_def_n = os.path.join(self.work_space,'class_definitions.csv')
        data = pd.read_csv(cls_def_n, sep=',', header='infer', usecols=[1])
        for i in range(data.size):
            cls = data.loc[i].Name
            self.name2label[cls] = len(self.name2label)
            self.label2class[len(self.label2class)] = cls
        return

    def parse_entries(self):
        cls_def_n = os.path.join(self.work_space, 'entries.csv')
        file_paths = pd.read_csv(cls_def_n, sep=',', header='infer', usecols=[1]).FilePath
        gts = pd.read_csv(cls_def_n, sep=',', header='infer', usecols=[6]).ClassIdxGroundTruth

        for i in range(len(self.label2class)):
            file_i = file_paths[gts == i]
            select_n = int(self.split_ratio*len(file_i))
            cls_i = self.label2class[i]
            self.name2path_train[i] = [self.work_space+ n for n in file_i[:select_n]]
            self.name2path_val[i] = [self.work_space+ n for n in file_i[select_n:]]
        return

    def mil_data_initial(self):
        #解析class_definitions
        self.parse_class_definitions()
        #解析entries
        self.parse_entries()
        #生成train_data/train_loader
        self.model_data_initial()



    def pytorch_data_initial(self):
        self.get_name2label()
        self.model_data_initial()

    def folder_initial(self):
        os.makedirs(self.model_dst_dir,exist_ok=True)

    def criteon(self, logits, y, class_ratio):
        # 设置二分类loss权重，alpha
        alpha_arr = list(class_ratio)
        alpha_list = torch.tensor([alpha_arr[x] for x in y]).reshape(-1, 1).to(self.calc_device)
        # # 变色：0；正常：1；残留：2；胶丝：3
        loss_log = torch.log(logits).to(self.calc_device)
        loss = torch.nn.functional.nll_loss(loss_log * alpha_list, y).to(self.calc_device)
        return loss

    def get_name2label(self):
        file_list = os.listdir(self.src_dir)
        for file in file_list:
            if '.' not in file:
                cls_dir = os.path.join(self.src_dir, file)
                self.name2label[file] = len(self.name2label)
                self.label2class[len(self.label2class)] = file
                self.name2path[file] = os.listdir(cls_dir)
        for k,v in self.name2path.items():
            random.shuffle(v)
            select_n = int(self.split_ratio*len(v))
            self.name2path_train[k] = [os.path.join(self.src_dir,k,n) for n in v[:select_n]]
            self.name2path_val[k] =   [os.path.join(self.src_dir,k,n) for n in v[select_n:]]
        return

    def model_data_initial(self):
        #划分训练数据以及，验证数据
        self.train_db = WaferImage(self.name2path_train,self.input_size,self.use_feature,self.use_mask)
        self.val_db = WaferImage(self.name2path_val,self.input_size,self.use_feature,self.use_mask)
        self.train_loader = DataLoader(self.train_db, batch_size=self.batchsz, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_db, batch_size=self.batchsz, num_workers=0)
        self.class_num_ratio =[len(cls)/len(self.train_db) for cls in self.name2path_train.values()]





if __name__ == '__main__':
    calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    para_test = Argu(calc_device)
