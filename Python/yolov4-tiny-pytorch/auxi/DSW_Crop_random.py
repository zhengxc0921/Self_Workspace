import cv2
import numpy as np
import os


# import torchvision.transforms as transforms


# 功能：从Random(0.36)上切割出适用于目标检测的小区，以及defect的box

class DefectPrepare:
    def __init__(self, DETECTION, project):
        self.SAVEDEFECTIMG = True
        self.DETECTION = DETECTION
        # 模板图路径
        self.project = project
        self.src_img_path = r'G:\DefectDataCenter\DSW\Random(0.36).bmp'
        self.dst_prefix = r'I:\MIL_Detection_Dataset\{}\raw_data'.format(project)
        self.dst_img_dir = self.dst_prefix + r'\img'
        self.dst_defect_dir = self.dst_prefix + r'\ClassesIcon'
        self.config_label_path = self.dst_prefix + r'\ImgBoxes.txt'
        self.config_classes_path = self.dst_prefix + r'\Classes.txt'

        # self.cell_size = [48, 89]
        # self.cell_sapce = [0, 9]
        # self.cell_period_y = self.cell_size[1] + self.cell_sapce[1]

        self.block_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

        self.block_name = None

        # self.crop_img_O_axis = [19633, 489]
        self.crop_img_O_axis = [19028, 489]
        self.block_step_y = [0, 3325, 3326, 3326, 3326, 3325, 3325, 3326, 3326]
        self.block_step_x = [0, 0, -2, 1, -2, -2, -1, -3, -1]
        # 同一个block内，缺陷cell的分布
        # self.cell_step_y =   [ 0,  245, 200-3,  448-7, 218-2, 290,270, 433,453, 220,260]
        self.cell_step_y = [0, 245, 200, 448 - 2, 218 - 2, 295, 270, 433, 453, 220, 261]
        # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J','K']
        self.cell_step_x = 906
        self.cell_size = [906, 300]  # 宽，高

        self.defect_step = [302, 0]

        ##基于目标检测的缺陷分布:每个defect在cell中的位置
        self.defect_center = {'A': [203, 127], 'B': [217, 135], 'C': [274, 121],
                              'D': [274, 122], 'E': [40, 125], 'F': [37, 120], 'G': [277, 108],
                              'H': [54, 100], 'I': [114, 109], 'J': [263, 108], 'K': [292, 118]}
        # H，I:该缺陷为粗细
        self.defect_szie = [20, 20]

        # self.crop_img_step_y = np.array([0, 294, 392, 490, 588, 490, 392, 294, 392, 588, 490, 294])  # 添加A-->A，step_y=0
        #                                    'A','B','C','D','E','F','G','H','I','J','K','L'
        # self.crop_img_step_y_cail = np.array([0, -4, -5, -5, -8, -5, -5, -4, -4, -9, -6, -4])  # 添加A-->A，step_y=0
        # self.crop_img_step_y += self.crop_img_step_y_cail
        # self.crop_img_step_x = np.arange(4) * 18 * self.cell_size[0]

        self.defect_type_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        self.defect_type_dict = dict()
        self.defect_mask = {'A': 3, 'B': 3, 'H': 4, 'I': 4, 'J': 5} #控制col前段
        self.defect_label = 0
        self.folder_init()

    def config_classes_init(self):
        with open(self.config_classes_path, 'w') as f:
            for class_type in self.defect_type_list:
                f.write(class_type)
                f.write('\n')
        f.close()

    def folder_init(self):

        if not os.path.exists(self.dst_img_dir):
            os.makedirs(self.dst_img_dir)
        if not os.path.exists(self.dst_defect_dir):
            os.makedirs(self.dst_defect_dir)

        dst_defect_paths = [os.path.join(self.dst_defect_dir, defect_type) for defect_type in self.defect_type_list]
        for dst_defect_path in dst_defect_paths:
            if not os.path.exists(dst_defect_path):
                os.makedirs(dst_defect_path)
        for name in self.defect_type_list:
            self.defect_type_dict[name] = len(self.defect_type_dict)
        self.config_classes_init()

    def crop_img(self, block_index, f):

        img_raw = cv2.imdecode(np.fromfile(self.src_img_path, dtype=np.uint8), -1)
        self.crop_img_start_y = self.block_start_y
        # self.crop_img_O_axis[0] = self.block_start_x
        for row_index, step_y in enumerate(self.cell_step_y):
            self.crop_img_start_y += step_y  # Y每次截取后坐标更新
            self.defect_type = self.defect_type_list[row_index]
            self.defect_label = row_index

            # 加入缺陷分布的map
            self.eff_col_index_list = np.arange(6)[::-1] + 1 ##控制col前段，设置col_1-col_6有效
            if self.defect_type in self.defect_mask:
                col_num = self.defect_mask[self.defect_type]
                self.eff_col_index_list = self.eff_col_index_list[:7 - col_num]
            self.mask_col_index = np.array((9, 8, 7))
            for step_x, col_index in enumerate(np.arange(3)):
                self.cell_col_index = step_x
                # self.crop_img_start_x = self.crop_img_O_axis[0] + step_x* self.cell_step_x
                self.crop_img_start_x = self.block_start_x + step_x * self.cell_step_x
                # X每次从crop_img_O_X处开始截取
                self.img_cut = img_raw[self.crop_img_start_y:self.crop_img_start_y + self.cell_size[1],
                               self.crop_img_start_x:self.crop_img_start_x + self.cell_size[0]].astype('int')
                # 存储某个Crop_Img
                crop_img_name = f'block_{self.block_name}_defect_{self.defect_type}_cellcol_{self.cell_col_index}.bmp'
                dst_img_path = os.path.join(self.dst_img_dir, crop_img_name)
                defect_box_list = self.save_defect()
                if(len(defect_box_list)>1):
                    cv2.imwrite(dst_img_path, self.img_cut)
                    f.write(dst_img_path)
                    f.write(defect_box_list)
                    f.write("\n")
                self.mask_col_index -= 3

    def save_defect(self):
        self.defect_step_x_list = [0, 604, 302]
        defect_box_list = ' '
        self.col_index = np.intersect1d(self.mask_col_index, self.eff_col_index_list)
        for step, col_index in enumerate(self.col_index):
            cell_step_x = self.defect_step_x_list[col_index % 3]
            for cell_index_y in [0]:
                cell_step_y = cell_index_y * self.defect_step[1]
                rela_center_x = self.defect_center[self.defect_type][0]
                rela_center_y = self.defect_center[self.defect_type][1]
                abs_center_x = self.crop_img_start_x * 0 + self.cell_size[0] * 0 + rela_center_x + cell_step_x
                abs_center_y = self.crop_img_start_y * 0 + self.cell_size[1] * 0 + rela_center_y + cell_step_y
                # defect_box : [x1,y1,x2,y2]
                defect_box = [abs_center_x - self.defect_szie[0] // 2, abs_center_y - self.defect_szie[1] // 2,
                              abs_center_x + self.defect_szie[0] // 2, abs_center_y + self.defect_szie[1] // 2]
                defect_box_tmp = " " + ",".join([str(a) for a in defect_box]) + "," + str(self.defect_label)
                defect_box_list = defect_box_list + defect_box_tmp

                if self.SAVEDEFECTIMG:
                    defect_img = self.img_cut[defect_box[1]:defect_box[1] + self.defect_szie[1],
                                 defect_box[0]:defect_box[0] + self.defect_szie[0]]
                    defect_name = f'block_{self.block_name}_defect_{self.defect_type}' \
                                  f'_cellrow_{cell_index_y}_col{col_index}.bmp'
                    dst_defect_path_tmp = os.path.join(self.dst_defect_dir, self.defect_type)
                    dst_defect_path = os.path.join(dst_defect_path_tmp, defect_name)
                    cv2.imwrite(dst_defect_path, defect_img)
        return defect_box_list[1:]

    def crop_all_block_(self):
        # 打开文件夹
        with open(self.config_label_path, 'w') as f:
            self.block_start_y = self.crop_img_O_axis[1]
            self.block_start_x = self.crop_img_O_axis[0]
            for block_index, block_name in enumerate(self.block_names):
                self.block_start_y += self.block_step_y[block_index]
                self.block_start_x += self.block_step_x[block_index]
                self.block_name = block_name
                # print(self.block_start_y)
                self.crop_img(block_index, f)

            f.close()


if __name__ == '__main__':
    DETECTION = 1  # 0:用于分类的截图；1：用于检测的截图
    project = 'DSW_random'
    test = DefectPrepare(DETECTION, project)
    test.crop_all_block_()
