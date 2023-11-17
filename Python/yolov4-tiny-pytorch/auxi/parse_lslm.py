import re,os
import cv2
class Parse:

    def __init__(self):
        self.parse_data_dir = r'G:/DefectDataCenter/ParseData/Detection/'
        self.general_data_dir = r'G:/DefectDataCenter/General_Data/'

    def parse_lslm(self):
        self.proj = 'lslm'

        self.src_train_dir = self.general_data_dir + self.proj + '/train/'
        self.src_val_dir = self.general_data_dir + self.proj + '/val/'
        self.src_train_path = self.src_train_dir + 'train.txt'
        self.src_val_path = self.src_val_dir + 'eval.txt'

        self.dst_dir = self.parse_data_dir + self.proj + '/raw_data'
        self.dst_train_path =  self.dst_dir + '/ImgBoxes_train.txt'
        self.dst_val_path =  self.dst_dir + '/ImgBoxes_val.txt'
        self.dst_classes_path =  self.dst_dir + '/Classes.txt'

        self.class_map = {'bolt': '0', 'nut': '1'}
        self.IconDir  = self.dst_dir + '/ClassesIcon/'
        self.IconWriteList = []
        self.parse_lslm_txt()

    def parse_lslm_txt(self):
        os.makedirs(self.IconDir ,exist_ok=True)
        with open(self.dst_train_path,'w') as dst_file:
            with open(self.src_train_path,'r') as src_file:
                frls = src_file.readlines()
                for rl in frls:
                    rl_sp = rl.split("\t")
                    img_path = self.src_train_dir +rl_sp[0]
                    dst_file.write(img_path)
                    for strl in rl_sp[1:]:
                        if len(strl)>1:
                            x1y1x2y2 = [float(s) for s in re.findall(r'-?\d+\.?\d*', strl)]
                            box_str =' '+','.join(str(x) for x in x1y1x2y2)+','
                            if "bolt" in strl:
                                class_type = 'bolt'
                            else:
                                class_type = 'nut'

                            dst_file.write(box_str+self.class_map[class_type])
                    dst_file.write('\n')

        with open(self.dst_val_path,'w') as dst_file:
            with open(self.src_val_path,'r') as src_file:
                frls = src_file.readlines()
                for rl in frls:
                    rl_sp = rl.split("\t")
                    img_path = self.src_val_dir +rl_sp[0]
                    dst_file.write(img_path)
                    for strl in rl_sp[1:]:
                        if len(strl)>1:
                            x1y1x2y2 = [float(s) for s in re.findall(r'-?\d+\.?\d*', strl)]
                            x1y1x2y2Int = [int(s) for s in x1y1x2y2]
                            box_str =' '+','.join(str(x) for x in x1y1x2y2)+','
                            if "bolt" in strl:
                                class_type = 'bolt'
                            else:
                                class_type = 'nut'
                            ##将Icon写入ClassIcon
                            if class_type not in self.IconWriteList:
                                Icon_path = self.IconDir + class_type+'.bmp'
                                img = cv2.imread(img_path)
                                Icon = img[x1y1x2y2Int[1]:x1y1x2y2Int[3],x1y1x2y2Int[0]:x1y1x2y2Int[2]]
                                cv2.imwrite(Icon_path,Icon)
                                self.IconWriteList.append(class_type)
                            dst_file.write(box_str+self.class_map[class_type])
                    dst_file.write('\n')

        with open(self.dst_classes_path,'w') as dst_file:
            for k,v in self.class_map.items():
                dst_file.write(k)
                dst_file.write('\n')

if __name__ == '__main__':
    A = Parse()
    A.parse_lslm()
