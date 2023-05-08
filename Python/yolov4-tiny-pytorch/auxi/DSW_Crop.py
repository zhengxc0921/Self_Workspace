import cv2
import numpy as np
import os


# import torchvision.transforms as transforms


# 功能：从Array(0.48)上切割出适用于目标检测的小区，以及defect的box

class DefectPrepare:
    def __init__(self, DETECTION, project):
        self.SAVEDEFECTIMG = True
        self.DETECTION = DETECTION
        # 模板图路径
        self.project = project
        self.src_img_path = r'G:\DefectDataCenter\DSW\Array(0.48).bmp'
        self.dst_prefix = r'I:\MIL_Detection_Dataset\{}\raw_data'.format(project)
        self.dst_img_dir = self.dst_prefix + r'\img'
        self.dst_defect_dir = self.dst_prefix + r'\ClassesIcon'

        # self.template_dir = self.dst_prefix + r'\template'
        # self.template_path = self.dst_prefix + r'\template\template.bmp'

        # self.config_dir = self.dst_prefix + r'\config'
        self.config_label_path = self.dst_prefix + r'\ImgBoxes.txt'
        self.config_classes_path = self.dst_prefix + r'\Classes.txt'

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
        # self.crop_template()
        self.folder_init()

    def config_classes_init(self):
        with open(self.config_classes_path, 'w') as f:
            for class_type in self.defect_type_list:
                f.write(class_type)
                f.write('\n')
        f.close()

    def defect_distribution_init(self):
        if self.DETECTION:
            ##基于目标检测的缺陷分布
            self.defect_center = {'A': [17, 9], 'B': [41, 45], 'C': [18, 81],
                                  'D': [29, 18], 'E': [40, 63], 'F': [33, 36], 'G': [37, 64],
                                  'H': [33, 24], 'I': [15, 15], 'J': [16, 14], 'K': [16, 56], 'L': [19, 70]}
            self.defect_szie = [30, 25]
        else:
            ##基于分类的缺陷分布
            # self.defect_center = {k:self.cell_size//2 for k in self.defect_type_list}
            self.defect_center = {k: [self.cell_size[0] // 2, self.cell_size[1] // 2] for k in self.defect_type_list}
            # self.defect_center = {'A': [17, 9], 'B': [41, 45], 'C': [18, 81],
            #                       'D': [29, 18], 'E': [40, 63], 'F': [33, 36], 'G': [37, 64],
            #                       'H': [33, 24], 'I': [15, 15], 'J': [16, 14], 'K': [16, 56], 'L': [19, 70]}
            self.defect_szie = self.cell_size

    def folder_init(self):
        # if not os.path.exists(self.template_dir):
        #     os.makedirs(self.template_dir)

        # if not os.path.exists(self.config_dir):
        #     os.makedirs(self.config_dir)

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
                dst_img_path = os.path.join(self.dst_img_dir, crop_img_name)
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

    # def crop_template(self):
    #     h_plus = 20
    #     w_plus = 20
    #     col_step = 2
    #     img_raw = cv2.imdecode(np.fromfile(self.src_img_path, dtype=np.uint8), -1)
    #     template = img_raw[1956 - h_plus:294 + 1956 + h_plus,
    #                21822 - w_plus - 720 * col_step:720 + 21822 + w_plus - 720 * col_step].astype('int')
    #     cv2.imwrite(self.template_path, template)

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
                    dst_defect_path_tmp = os.path.join(self.dst_defect_dir, self.defect_type)
                    dst_defect_path = os.path.join(dst_defect_path_tmp, defect_name)
                    cv2.imwrite(dst_defect_path, defect_img)
                    ##保存模板圖
                    # img_copy = np.zeros_like(self.img_cut)
                    # img_copy[defect_box[1]:defect_box[1] + self.defect_szie[1],
                    # defect_box[0]:defect_box[0] + self.defect_szie[0]] = 1
                    # dst_mask_path = os.path.join(self.dst_img_dir, defect_name)
                    # cv2.imwrite(dst_mask_path, img_copy)

        return defect_box_list[1:]

    def crop_all_block_(self):
        # 打开文件夹
        with open(self.config_label_path, 'w') as f:
            self.crop_img_start_y = self.crop_img_O_axis[1]
            for block_index, block_name in enumerate(self.block_names):
                self.block_name = block_name
                self.crop_img_start_y += self.block_step_y[block_index]
                # self.crop_img_start_y +=  block_index*(self.block_step_y[1])
                self.crop_img(block_index, f)
            f.close()


if __name__ == '__main__':
    DETECTION = 1  # 0:用于分类的截图；1：用于检测的截图
    project = 'DSW'
    test = DefectPrepare(DETECTION, project)
    test.crop_all_block_()
