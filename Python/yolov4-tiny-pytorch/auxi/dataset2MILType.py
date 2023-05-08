# import matplotlib.pyplot as plt
import random
import string
import shutil
import os,csv,cv2
import numpy as np
class CSV2MILDataSet:
    def __init__(self):
        self.proj = 'DSW'
        self.wksp = self.proj+"//" +"Dataset"
        self.class_num = 0
        self.img_num = 0
        self.Authors = 'ZXC'
        self.Weight = '1'

        self.img_UUID = ''
        self.class_UUID = ''
        self.src_Icon_dir = "../model_input/{}/raw_data/defect".format(self.proj)
        self.src_dir = "../model_input/{}/raw_data/config".format(self.proj)
        self.src_classes_def = os.path.join(self.src_dir,'classes.txt')
        self.src_boxes_def =  os.path.join(self.src_dir,'train_box_raw.txt')
        self.classes = []
        self.img_box_map = {}
        self.dst_Icon_dir = self.proj+'/Dataset/Icons/'
        self.dst_Image_dir =self.proj+'/Dataset/Images/'


    def initial_workfolder(self):
        os.makedirs(self.dst_Icon_dir, exist_ok=True)
        os.makedirs(self.dst_Image_dir, exist_ok=True)
        self.read_src_classes_def()
        self.gen_Icon_mim()

        self.read_src_boxes_def()


    def read_src_classes_def(self):
        with open(self.src_classes_def, 'r') as f:
            for class_n in f.readlines():
                self.classes.append(class_n.strip())
        self.class_num = len(self.classes)


    def gen_Icon_mim(self):
        #遍历所有类别
        for cls in self.classes:
            src_Icon_n = os.path.join(self.src_Icon_dir,cls)
            src_Icon_path =  os.path.join(src_Icon_n,os.listdir(src_Icon_n)[0])
            dst_Icon_path = os.path.join(self.dst_Icon_dir,cls+'.mim')
            shutil.copy(src_Icon_path,dst_Icon_path)

    def read_src_boxes_def(self):

        with open(self.src_boxes_def, 'r') as f:
            for img_box_str in f.readlines():
                img_box_list = img_box_str.strip().split(' ')
                img_box_len = len(img_box_list)
                #['path','box label'] --->['path':['box1 label1','box2 label2'....]]
                if img_box_len>1:
                    #完成图片的复制src_img-->dst_img
                    src_Image_path = img_box_list[0]
                    dst_Image_path = self.dst_Image_dir +src_Image_path.split("\\")[-1]
                    shutil.copy(src_Image_path, dst_Image_path)
                    #绑定dst_Image_path对应的box label
                    key_img ='Images/'+ src_Image_path.split("\\")[-1]
                    self.img_box_map[key_img] = [[img_box_list[1]]]
                    for i in range(2,img_box_len):
                        self.img_box_map[key_img].append([img_box_list[i]])
        self.img_num = len(self.img_box_map)
        return





    def general_UUID(self,num,use_ID):
        UUID_tail = '-7AF8-12EC-BEB{}-646C809A1416'.format(use_ID)
        UUID_list = []
        for i in range(num):
            str_list = random.sample(string.digits + string.ascii_uppercase, 8)
            UUID_head = ''.join(str_list)
            UUID = UUID_head+UUID_tail
            UUID_list.append(UUID)


        return UUID_list


    def gen_authors(self):
        row_list = [["Key","Name"]]
        row_next = [self.general_UUID(1,0)[0],self.Authors]
        row_list.append(row_next)
        authors_path = os.path.join(self.wksp,'authors.csv' )
        with open(authors_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


    def gen_class_definitions(self):
                    # ***	class_n   238         44         44     ***\Name.mim   1
        row_list = [['Key',	'Name',	'Color_R',	'Color_G',	'Color_B',	'Icon',	'Weight']]
        RGB = ['238','44','44']
        #读取src classes definition
        self.class_UUID = self.general_UUID(self.class_num,1)
        for i,cls in enumerate(self.classes):
            row = []
            row.append(self.class_UUID[i])
            row.extend(cls)
            row.extend(RGB)
            row.append('Icons/'+cls+'.mim')
            row.extend(self.Weight)
            row_list.append(row)
        class_definitions_path = os.path.join(self.wksp,'class_definitions.csv' )
        with open(class_definitions_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


    def gen_controls(self):
        row_list = [["ControlType", "ControlValue"],
                    ["M_NO_CLASS_DRAW_COLOR", "M_COLOR_NO_CLASS"],
                    ["M_NO_REGION_PIXEL_CLASS", "M_NO_CLASS"],
                    ["M_DONT_CARE_CLASS_DRAW_COLOR", "M_COLOR_DONT_CARE_CLASS"]]

        controls_path = os.path.join(self.wksp,'controls.csv' )
        with open(controls_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

    def gen_descriptor_boxes(self):
        row_list = [['Key',	'FilePath',	'RegionIndex',	'ClassIdxGroundTruth',	'AuthorName',	'BoxCoords']]
        i = -1
        for  img_path,boxes in self.img_box_map.items():
            i +=1
            for j,box_label in enumerate(boxes):
                box,label = ','.join(box_label[0].split(',')[:-1]),box_label[0].split(',')[-1]
                row = []
                row.append(self.img_UUID[i])
                row.append(img_path)
                row.append(j+1)
                row.append(int(label))
                row.append(self.Authors)
                row.append(box)
                row_list.append(row)

        descriptor_boxes_path = os.path.join(self.wksp,'descriptor_boxes.csv' )
        with open(descriptor_boxes_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)



    def gen_entries(self):
        row_list = [['Key','FilePath','AuthorName',
        'AugmentationSource',
        'ExternalAugSource',
        'RegionType',
        'ClassIdxGroundTruth',
        'UserString',
        'UserConfidence']]

        self.img_UUID = self.general_UUID(self.img_num,2)
        for i,img_path in enumerate(self.img_box_map):
            row = []
            row.append(self.img_UUID[i])
            row.append(img_path)
            row.append(self.Authors)
            row.append('NOT_AUGMENTED')
            row.append('00000000-0000-0000-0000-000000000000')
            row.append('WholeImage')
            row.extend(' ')
            row.extend(' ')
            row.extend('0')
            row_list.append(row)

        entries_path = os.path.join(self.wksp,'entries.csv' )
        with open(entries_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        return

    def csv2mildata(self):
        self.initial_workfolder()
        #workspace 下依次建立以下csv
        #1、创建 authors.csv
        self.gen_authors()
        #2、创建class_definitions.csv
        self.gen_class_definitions()
        #3、创建controls.csv
        self.gen_controls()

        #4、创建entries.csv
        self.gen_entries()

        #5、创建descriptor_boxes.csv
        self.gen_descriptor_boxes()


if __name__ == '__main__':

    a = CSV2MILDataSet()
    a.csv2mildata()