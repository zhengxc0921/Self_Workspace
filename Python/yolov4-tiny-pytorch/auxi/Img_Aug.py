import os
import cv2
import numpy as np
import shutil

class ImgAug:

    def __init__(self):
        pass

    def broad_img(self):
        src_dir = r'G:\DefectDataCenter\ParseData\Detection\DSW_random\raw_data\TImg_bk'
        dst_dir = r'G:\DefectDataCenter\ParseData\Detection\DSW_random\raw_data\TImg'

        gt_img = np.ones((1080,1440)) ##hw
        src_list = os.listdir(src_dir)
        for src_img in src_list:
            src_path = os.path.join(src_dir ,src_img)
            dst_path = os.path.join(dst_dir ,src_img)
            src_img = cv2.imread(src_path,-1)
            h,w = src_img.shape
            gt_img[:h,:w] = src_img

            cv2.imwrite(dst_path,gt_img)

    def copy_folder(self,src,dst):
        os.makedirs(dst,exist_ok=True)
        src_files = os.listdir(src)
        for file in src_files :
            if '.' not in file:
                src_folder= os.path.join(src,file)
                dst_folder = os.path.join(dst,file)
                self.copy_folder(src_folder,dst_folder)
            else:
                src_file = os.path.join(src, file)
                dst_file = os.path.join(dst, file)
                shutil.copy(src_file,dst_file)


    def parseTxt_JPG2BMP(self,src_txt,dst_path):
        with open(dst_path,'w') as dstf:
            with open(src_txt, 'r') as srcf:
                fls = srcf.readlines()
                img_path = fls[0].split(' ')[0]
                src_dir_path = os.path.dirname(img_path)
                dst_path = src_dir_path.replace('lslm', 'lslm_bmp')
                os.makedirs(dst_path, exist_ok=True)

                for fl in fls:
                    dst_line = fl.replace('lslm', 'lslm_bmp').replace('jpg', 'bmp')
                    src_img_path = fl.split(' ')[0]
                    dst_img_path = dst_line.split(' ')[0]
                    img = cv2.imread(src_img_path)
                    cv2.imwrite(dst_img_path, img)
                    dstf.write(dst_line)




    def parse_JPG2BMP(self):
        ##将lslm数据集3通道的jpg解压成bmp
        src_dir = r'G:\DefectDataCenter\ParseData\Detection\lslm'
        dst_dir = r'G:\DefectDataCenter\ParseData\Detection\lslm_bmp'
        #复制文件
        self.copy_folder(src_dir,dst_dir)
        #读取src_dir 中的文件
        src_train_txt =  os.path.join(src_dir,'raw_data','ImgBoxes_train.txt')
        src_val_txt = os.path.join(src_dir, 'raw_data', 'ImgBoxes_val.txt')

        dst_train_txt = os.path.join(dst_dir, 'raw_data', 'ImgBoxes_train.txt')
        dst_val_txt = os.path.join(dst_dir, 'raw_data', 'ImgBoxes_val.txt')
        self.parseTxt_JPG2BMP(src_train_txt,dst_train_txt)
        self.parseTxt_JPG2BMP(src_val_txt,dst_val_txt)


    # def













if __name__ == '__main__':

    A = ImgAug()
    A.broad_img()
