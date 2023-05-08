import cv2
import os
import numpy as np

class HWCroper:
    def __init__(self):
        project = 'HW'
        self.ClassName = "A"
        self.dft_size = [15,15]

        self.src_img_dir = r'G:\DefectDataCenter\华为数据\检测数据'
        self.dst_dir = r'I:\MIL_Detection_Dataset\{}\raw_data'.format(project)
        # self.dst_config_dir = os.path.join(self.dst_dir,r'config')
        # self.dst_img_dir = os.path.join(self.dst_dir,'Img')
        self.dst_defect_dir = os.path.join(self.dst_dir,'ClassesIcon')
        self.dst_ImgBoxes_path =  os.path.join(self.dst_dir,'ImgBoxes.txt')
        self.dst_Classes_path = os.path.join(self.dst_dir, 'Classes.txt')
        self.folder_initial()



    def folder_initial(self):
        #初始化结果保存文件夹

        if not os.path.exists(self.dst_defect_dir):
            os.makedirs(self.dst_defect_dir)



    def HW_crop(self):
        with open(self.dst_Classes_path,'w') as Classes_file:
            Classes_file.write(self.ClassName)

        with open(self.dst_ImgBoxes_path,'w') as config_file:
            for img_name in os.listdir(self.src_img_dir):
                img_path = os.path.join(self.src_img_dir,img_name)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                height,width = img.shape
                ##[x1,y1,x2,y2]
                dft_box = [width//2-self.dft_size[1]//2,height//2-self.dft_size[0]//2,
                              width // 2 + self.dft_size[1] // 2, height // 2 + self.dft_size[0] // 2
                              ]
                ##保存缺陷
                defect = img[dft_box[1]:dft_box[3],dft_box[0]:dft_box[2]]
                defect_name = os.path.join(self.dst_defect_dir,self.ClassName+'.bmp')
                cv2.imwrite(defect_name,defect)
                ##图片名字，以及defect_box ,type写入config——file
                string = img_path+" "+ ",".join(str(a) for a in dft_box)+",0\n"
                config_file.write(string)
        # config_file.close()





if __name__ == '__main__':
    test = HWCroper()
    test.HW_crop()