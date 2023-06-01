#!/usr/bin/env Python
# coding=utf-8
import torch
import os
import scipy.io as io

import cv2
import random
import numpy as np

from torch.utils.data import Dataset
class WaferImage(Dataset):

    def __init__(self, name2path,img_size,use_feature,use_mask):
        super(WaferImage, self).__init__()
        self.name2path = name2path
        self.img_size = img_size
        self.label2class = {}
        self.use_feature = use_feature
        self.use_mask = use_mask
        self.blob_root = r'G:\DefectDataCenter\DeepLearningDataSet\Input'
        self.extra_dir = 'DXC_feature'
        self.extra_files = ['DXC_result.mat','blob_attribute_list.mat','B_list.mat']
        self.class_num_ratio = []
        self.class_nums = []
        self.class_num = 0
        self.parap_initial()

    def __len__(self):
        return len(self.img_lbs_list)

    def parap_initial(self):
        #展开name2path
        self.img_lbs_list =[]

        for k,v in self.name2path.items():
            self.img_lbs_list.extend([img_path + ',' +str(k) for img_path in v])
        random.shuffle(self.img_lbs_list)
        self.normal_blob = torch.zeros(0, dtype=torch.float32)  #use_feature false的时候占位用
        if self.use_feature:
            self.load_mat_feature()
            self.export_blobs_excel()

    def export_blobs_excel(self):
        # import torch
        n_b_list = []
        for i in range(len(self.normal_blobs)):
            n_b = self.normal_blobs[i]
            if n_b.sum()==0:
                ##normal ,normal的label为1
                n_b_p = np.array(torch.cat((n_b,torch.tensor([1]))))
            else:
                ##defect，defect的label为0
                n_b_p = np.array(torch.cat((n_b, torch.tensor([0]))))
            n_b_list.append(n_b_p)
        np.savetxt("../blobs.csv", np.array(n_b_list), delimiter=",")

    def load_mat_feature(self):
        self.f_len = 8
        self.Blobs_path = os.path.join(self.blob_root, self.extra_dir, self.extra_files[1])
        self.Blobs = io.loadmat(self.Blobs_path)
        self.blobs_list = io.loadmat(self.Blobs_path)['blob_attribute_list'][0]
        self.blobs_len = len(self.blobs_list)
        self.normal_blobs = torch.zeros([self.blobs_len, self.f_len], dtype=torch.float32)
        self.normalize_blobs()

    def normalize_blobs(self):
        cx = cy = 250
        r2 = 300
        ##遍历图片
        for i in range(self.blobs_len):
            blobs = self.blobs_list[i]
            if bool(blobs.shape[0]):
                #查询blob的px.py，进而确认有效的blob
                for j in range(blobs.shape[1]):
                    blob = blobs[0][j]
                    px,py = blob[0][0][0],blob[1][0][0]
                    if (px-cx)**2+(py-cy)**2<r2:
                        po = blob[2][0][0]
                        bt = blob[3][0][0]/255
                        br = blob[4][0][0]/255
                        rt = blob[5][0][0]/255
                        rr = blob[6][0][0]/255
                        m_gl = blob[7][0][0]/255
                        m_mag = blob[8][0][0]/255
                        m_oft = blob[9][0][0]/255
                        m_spt_like = blob[10][0][0]/255
                        energy = blob[11][0][0]/255
                        self.normal_blobs[i] = torch.tensor([po,br,rr,m_gl,m_mag,m_oft,m_spt_like,energy],dtype=torch.float32)
                        break
        return

    def cv_load_image(self, img_read):
        # 灰度图转换为彩图
        if len(img_read.shape)==3:
            img_cv_gray = img_read/255.0
        else:
            img_cv_gray = cv2.cvtColor(img_read,cv2.COLOR_GRAY2BGR)/255.0
        #debug  ,查看中间图片是否正常
        # plt.ion()
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(img_cv_gray[:, :, 0] * 255)
        # plt.title('raw_img')
        # plt.subplot(1, 3, 2)
        # plt.imshow(self.Boundary_temp[:, :, 0] / 1.5)
        # plt.title('weight_mask')
        # plt.subplot(1, 3, 3)
        # plt.imshow(img_cv_gray[:, :, 0] * 1 + self.Boundary_temp[:, :, 0] * 0.1)
        # plt.title('raw_img+weight_mask')
        # plt.ioff()
        img_cv_resize = cv2.resize(img_cv_gray, self.img_size, interpolation=cv2.INTER_LINEAR)
        # # 图像裁剪之后的结果
        img_cv_resize_np_f64 = np.transpose(img_cv_resize, [2, 0, 1])
        img_cv_resize_np = np.array(img_cv_resize_np_f64, dtype='float32')
        img_tensor = torch.tensor(img_cv_resize_np)
        return img_tensor

    def get_affiliated_info(self,idx):

        images_path, label = self.img_lbs_list[idx].split(",")
        img_read = cv2.imdecode(np.fromfile(images_path, dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        label = torch.tensor(int(label))
        image_ID = self.img_lbs_list[idx].split(os.sep)[-1].split('_')[0]
        ##label !=0 代表正常,设置feature=0
        if self.use_feature:
            self.normal_blob = self.normal_blobs[int(image_ID) - 1]
        return img_read,label,image_ID, images_path, self.normal_blob

    def __getitem__(self, idx):
        img_read,label, image_ID, images_path,normal_blob = self.get_affiliated_info(idx)
        # #label必须是defect的
        img = self.cv_load_image(img_read)
        return img, label, image_ID, images_path,normal_blob



if __name__ == '__main__':
    pass
    # root = r'G:\DefectDataCenter\DeepLearningDataSet\Input\Si'
    # mode = 'train'
    # datebase = WaferImage(root, mode)
    # train_loader = DataLoader(datebase, batch_size=64, shuffle=True,num_workers=0)
    # for step,(x,y,img_id,image_path,normal_blob) in  enumerate(train_loader):
    #     print(step)

