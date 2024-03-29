import cv2
import numpy as np
import os,random,shutil


# import torchvision.transforms as transforms
# 功能：从Array(0.48)上切割出适用于目标检测的小区，以及defect的box

class DefectPrepare:
    def __init__(self, DETECTION, proj):
        self.SAVEDEFECTIMG = True
        self.DETECTION = DETECTION
        # 模板图路径
        self.proj = proj
        self.src_img_path = r'G:\DefectDataCenter\原始_现场分类数据\DSW\Array(0.48).bmp'
        self.img_raw = cv2.imdecode(np.fromfile(self.src_img_path, dtype=np.uint8), -1)
        detection_root = r'G:\DefectDataCenter\ParseData\Detection'
        self.train_percent = 0.8
        self.cell_size = [48, 89]
        self.cell_sapce = [0, 9]
        self.cell_period_y = self.cell_size[1] + self.cell_sapce[1]
        self.block_names = ['A', 'B', 'C', 'D', 'E']
        self.block_name = None
        self.block_step_y = [0, 775, 776, 776, 776]
        self.block_step_x = [0, -2, -3, -6, -9]
        self.crop_img_O_axis = [21006 - 48, 1956]
        self.crop_img_szie = [5 * 2 * self.cell_size[0] + 5 * self.cell_size[0],
                              3 * (self.cell_sapce[1] + self.cell_size[1])]
        self.crop_img_step_y = np.array([0, 294, 392, 490, 588, 490, 392, 294, 392, 588, 490, 294])  # 添加A-->A，step_y=0
        #                                    'A','B','C','D','E','F','G','H','I','J','K','L'
        self.crop_img_step_y_cail = np.array([0, -4, -5, -5, -8, -5, -5, -4, -4, -9, -6, -4])  # 添加A-->A，step_y=0
        self.crop_img_step_y += self.crop_img_step_y_cail
        self.crop_img_step_x = np.arange(4) * 18 * self.cell_size[0]

        self.defect_type_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        self.defect_type_dict = dict()
        self.defect_mask = {'A': 3, 'K': 2, 'L': 3}
        self.defect_label = 0
        self.defect_step = [6 * self.cell_size[0], 0]
        self.defect_distribution_init()
        self.folder_initial(detection_root, proj)

    def folder_initial(self,detection_root, proj):
        #dst dir
        self.box_files = ['Train.txt', 'Val.txt']  # 提取xml ，生成原始的带训练数据
        self.saveConfigDir = r'{}\{}\raw_data\Config'.format(detection_root, proj)
        self.TrainValDataInfoPath = "{}\TrainVal.txt".format(self.saveConfigDir)
        self.saveDefectDir =  r'{}\{}\raw_data\defect'.format(detection_root, proj)
        self.saveImgDir =r'{}\{}\raw_data\Img'.format(detection_root, proj)
        self.saveTImgDir = r'{}\{}\raw_data\TImg'.format(detection_root, proj)
        self.saveClassIconDir = r'{}\{}\raw_data\ClassesIcon'.format(detection_root, proj)
        self.ini_path = os.path.join(self.saveConfigDir, "{}_Para.ini").format(proj)
        os.makedirs(self.saveConfigDir, exist_ok=True)
        os.makedirs(self.saveDefectDir, exist_ok=True)
        os.makedirs(self.saveImgDir, exist_ok=True)
        os.makedirs(self.saveTImgDir, exist_ok=True)
        os.makedirs(self.saveClassIconDir, exist_ok=True)
        for v in self.defect_type_list:
            dft_folder = os.path.join( self.saveDefectDir,v)
            os.makedirs(dft_folder,exist_ok=True)
        #输出Classes.txt
        with open(os.path.join(self.saveConfigDir, "Classes.txt"), 'w') as f:
            for v in self.defect_type_list:
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
            f.write("ImageSizeX={}\n".format(self.crop_img_szie[0]))
            f.write("ImageSizeY={}\n".format(self.crop_img_szie[1]))
            f.write("AugFreq=0\n")
            f.write("TestDataRatio=10\n")
        return

    def defect_distribution_init(self):
        if self.DETECTION:
            ##基于目标检测的缺陷分布
            self.defect_center = {'A': [17, 9], 'B': [41, 45], 'C': [18, 81],
                                  'D': [29, 18], 'E': [40, 63], 'F': [33, 36], 'G': [37, 64],
                                  'H': [33, 24], 'I': [15, 15], 'J': [16, 14], 'K': [16, 56], 'L': [19, 70]}
            self.defect_szie = [30, 25]
        else:
            ##基于分类的缺陷分布
            self.defect_center = {k: [self.cell_size[0] // 2, self.cell_size[1] // 2] for k in self.defect_type_list}
            self.defect_szie = self.cell_size

    def splitFile(self):
        #将Train_Val.txt拆分为Train.txt 和 Val.txt
        with open(self.TrainValDataInfoPath, 'r') as src_f:
            lines = src_f.readlines()
            random.shuffle(lines)
            train_num = int(self.train_percent*len(lines))
            train_lines = lines[:train_num]
            val_lines = lines[train_num:]
            for lines,file_n in zip([train_lines,val_lines],self.box_files):
                file_path = os.path.join(self.saveConfigDir,file_n)
                with open(file_path,'w') as f:
                    for line in lines:
                        f.write(line)

    def generalTestImg(self):
        ValInfo_path = "{}\Val.txt".format(self.saveConfigDir)
        with open(ValInfo_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                src_img_p = line.split(" ")[0]
                img_n = os.path.basename(src_img_p)
                dst_p  = os.path.join(self.saveTImgDir,img_n)
                shutil.copy(src_img_p,dst_p)

    def crop_img(self, block_index, f):
        img_raw = cv2.imdecode(np.fromfile(self.src_img_path, dtype=np.uint8), -1)
        for row_index, step_y in enumerate(self.crop_img_step_y):
            self.crop_img_start_y += step_y  # Y每次截取后坐标更新
            self.defect_type = self.defect_type_list[row_index]
            self.defect_label = row_index
            # 加入缺陷分布的map
            if self.defect_type in self.defect_mask:
                col_num = self.defect_mask[self.defect_type]
                crop_img_step_x_tmp = self.crop_img_step_x[1:col_num]
            else:
                crop_img_step_x_tmp = self.crop_img_step_x[1:]

            for col_index, step_x in enumerate(crop_img_step_x_tmp):
                self.col_index = col_index
                self.crop_img_start_x = self.crop_img_O_axis[0] + step_x + self.block_step_x[
                    block_index]  # X每次从crop_img_O_X处开始截取
                self.img_cut = img_raw[self.crop_img_start_y:self.crop_img_start_y + self.crop_img_szie[1],
                               self.crop_img_start_x:self.crop_img_start_x + self.crop_img_szie[0]].astype('int')
                # 存储某个Crop_Img
                crop_img_name = f'block_{self.block_name}_defect_{self.defect_type}_col_{self.col_index}.bmp'
                dst_img_path = os.path.join(self.saveImgDir, crop_img_name)
                cv2.imwrite(dst_img_path, self.img_cut)
                # 写入获取img的头部绝对路径
                # crop_img_head_abspath = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
                # crop_img_body_abspath = dst_img_path[3:]
                # crop_img_abspath = os.path.join(crop_img_head_abspath, crop_img_body_abspath)
                # storing defect img for classification test
                defect_box_list = self.save_defect()
                f.write(dst_img_path)
                f.write(defect_box_list)
                f.write("\n")

    def save_defect(self):
        defect_box_list = ' '
        for cell_index_x in range(3):
            cell_step_x = cell_index_x * self.defect_step[0]
            for cell_index_y in [0]:
                cell_step_y = cell_index_y * self.defect_step[1]
                rela_center_x = self.defect_center[self.defect_type][0]
                rela_center_y = self.defect_center[self.defect_type][1]
                abs_center_x = self.crop_img_start_x * 0 + self.cell_size[0] + rela_center_x + cell_step_x
                abs_center_y = self.crop_img_start_y * 0 + self.cell_size[1] + rela_center_y + int(
                    1.5 * self.cell_sapce[
                        1]) + cell_step_y
                # defect_box : [x1,y1,x2,y2]
                defect_box = [abs_center_x - self.defect_szie[0] // 2, abs_center_y - self.defect_szie[1] // 2,
                              abs_center_x + self.defect_szie[0] // 2, abs_center_y + self.defect_szie[1] // 2]
                defect_box_tmp = " " + ",".join([str(a) for a in defect_box]) + "," + str(self.defect_label)
                defect_box_list = defect_box_list + defect_box_tmp

                if self.SAVEDEFECTIMG:
                    defect_img = self.img_cut[defect_box[1]:defect_box[1] + self.defect_szie[1],
                                 defect_box[0]:defect_box[0] + self.defect_szie[0]]
                    defect_name = f'block_{self.block_name}_defect_{self.defect_type}_imgcol_' \
                                  f'{self.col_index}_cellrow_{cell_index_y}_cellcol{cell_index_x}.bmp'
                    dst_defect_path_tmp = os.path.join(self.saveDefectDir, self.defect_type)
                    dst_defect_path = os.path.join(dst_defect_path_tmp, defect_name)
                    cv2.imwrite(dst_defect_path, defect_img)
                    # 保存缺陷/存入Icon
                    ClassIcon_p = os.path.join(self.saveClassIconDir, self.defect_type + ".bmp")
                    if not os.path.exists(ClassIcon_p):
                        cv2.imwrite(ClassIcon_p, defect_img)
        return defect_box_list[1:]

    def generalTrainInfo(self):
        # 打开文件夹
        with open(self.TrainValDataInfoPath, 'w') as f:
            self.crop_img_start_y = self.crop_img_O_axis[1]
            for block_index, block_name in enumerate(self.block_names):
                self.block_name = block_name
                self.crop_img_start_y += self.block_step_y[block_index]
                self.crop_img(block_index, f)
            f.close()


if __name__ == '__main__':
    DETECTION = 1  # 0:用于分类的截图；1：用于检测的截图
    project = 'DSW'
    test = DefectPrepare(DETECTION, project)
    test.generalTrainInfo()
    test.splitFile()
    test.generalTestImg()
