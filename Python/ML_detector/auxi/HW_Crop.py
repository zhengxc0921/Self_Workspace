import shutil

import cv2
import os,random
import numpy as np

class HWCroper:
    def __init__(self):
        self.proj = 'HW'
        self.ClassName = "A"
        self.Croped_img = [224,224]
        self.dft_size = [15,15]
        ##Info from LabelImged
        self.img_dir = r'G:\DefectDataCenter\原始_现场分类数据\HuaWei'
        detection_root = r'G:\DefectDataCenter\ParseData\Detection'

        #general pytorch info
        self.clsMap = ['A']
        self.train_percent = 0.8
        self.folder_initial(detection_root, self.proj)

    def folder_initial(self,detection_root, proj):
        #dst dir
        self.box_files = ['Train.txt', 'Val.txt']  # 提取xml ，生成原始的带训练数据
        self.saveConfigDir = r'{}\{}\raw_data\Config'.format(detection_root, proj)
        self.saveDefectDir =  r'{}\{}\raw_data\defect'.format(detection_root, proj)
        self.saveCOTCropImgDir =r'{}\{}\raw_data\Img'.format(detection_root, proj)
        self.saveTImgDir = r'{}\{}\raw_data\TImg'.format(detection_root, proj)
        self.saveClassIconDir = r'{}\{}\raw_data\ClassesIcon'.format(detection_root, proj)
        self.ini_path = os.path.join(self.saveConfigDir, "{}_Para.ini").format(proj)

        os.makedirs(self.saveConfigDir, exist_ok=True)
        os.makedirs(self.saveDefectDir, exist_ok=True)
        os.makedirs(self.saveCOTCropImgDir, exist_ok=True)
        os.makedirs(self.saveTImgDir, exist_ok=True)
        os.makedirs(self.saveClassIconDir, exist_ok=True)
        for v in self.clsMap:
            dft_folder = os.path.join( self.saveDefectDir,v)
            os.makedirs(dft_folder,exist_ok=True)
        #输出Classes.txt
        with open(os.path.join(self.saveConfigDir, "Classes.txt"), 'w') as f:
            for v in self.clsMap:
                f.write(v + "\n")
        #输出ini文件
        self.gen_MILDataSetParaI_ini()

    def gen_MILDataSetParaI_ini(self ):
        with open(self.ini_path, 'w') as f:
            f.write("[{}]\n".format(self.proj))
            f.write("ClassesPath={}\Classes.txt\n".format(self.saveConfigDir))
            IconDir = self.saveConfigDir.replace("Config", '')
            f.write("IconDir={}ClassesIcon\ \n".format(IconDir))
            # f.write("TrainDataInfoPath={}\ImgBoxes_MIL_train.txt\n".format(self.saveConfigDir))
            f.write("TrainDataInfoPath={}\Train.txt\n".format(self.saveConfigDir))
            f.write("ValDataInfoPath={}\Val.txt\n".format(self.saveConfigDir))
            MILDir = IconDir.replace('raw_data', 'MIL_Data')
            f.write("WorkingDataDir={} \n".format(MILDir))
            f.write("PreparedDataDir={}PreparedData\n".format(MILDir))
            f.write("ImageSizeX={}\n".format(self.Croped_img[1]))
            f.write("ImageSizeY={}\n".format(self.Croped_img[0]))
            f.write("AugFreq=0\n")
            f.write("TestDataRatio=10\n")
        return

    def get_train_val_img(self):
        total_img = os.listdir(self.img_dir)
        # total_xml = [xml for xml in total_xml if xml.endswith('xml')]
        random.shuffle(total_img)
        train_num = int(len(total_img) * self.train_percent)
        train_img = total_img[:train_num]
        val_img = total_img[train_num:]
        return train_img,val_img

    def generalTestImg(self):
        ValInfo_path = "{}\Val.txt".format(self.saveConfigDir)
        with open(ValInfo_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                src_img_p = line.split(" ")[0]
                img_n = os.path.basename(src_img_p)
                dst_p  = os.path.join(self.saveTImgDir,img_n)
                shutil.copy(src_img_p,dst_p)

    def HW_crop(self):
        train_imgs, val_imgs = self.get_train_val_img()
        for imgs,file_n in zip([train_imgs,val_imgs], self.box_files):
            dst_path =os.path.join(self.saveConfigDir,file_n)
            with open(dst_path,'w') as config_file:
                for img_n in imgs:
                    img_path = os.path.join(self.img_dir,img_n)
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                    height,width = img.shape
                    ##[x1,y1,x2,y2]
                    dft_box = [width//2-self.dft_size[1]//2,height//2-self.dft_size[0]//2,
                                  width // 2 + self.dft_size[1] // 2, height // 2 + self.dft_size[0] // 2
                                  ]
                    ##图片名字，以及defect_box ,type写入config——file
                    img_box_info = img_path+" "+ ",".join(str(a) for a in dft_box)+",0\n"
                    config_file.write(img_box_info)

                    # 保存缺陷/存入Icon
                    ClassIcon_p = os.path.join(self.saveClassIconDir, self.clsMap[0] + ".bmp")
                    if not os.path.exists(ClassIcon_p):
                        dft = img[dft_box[1]:dft_box[3], dft_box[0]:dft_box[2]]
                        defect_name = os.path.join(self.saveDefectDir, self.ClassName + '.bmp')
                        cv2.imwrite(ClassIcon_p, dft)
                        cv2.imwrite(defect_name, dft)



if __name__ == '__main__':
    test = HWCroper()
    # test.HW_crop()
    test.generalTestImg()