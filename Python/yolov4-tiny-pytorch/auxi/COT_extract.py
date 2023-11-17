# 完成时间：2023-09-27
# 标签文件XML位于xmlfileDir： COT_Raw进行0.4*0.4缩放后标注的
# COT_Raw图片位于COT_RawDir：要将COT_Raw_resize的xml标签还原到COT_Raw图片上
# 根据还原的标签对COT_Raw进去裁剪
# 自适应裁剪得到 训练图片，以及defect
# Step1：解析COT 原始xml文件，生成train_box/val_box
# Step2:根据train_box/val_box按照缺陷位置进行裁剪出小图，生成ImgBoxes_train、ImgBoxes_val（用于pytorch训练）
# COT_Raw_Para.ini用于MIL训练
import os,cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
class ExtractResizeCOTXML2Raw:
    def __init__(self,proj,Croped_img_Size,Vaild_Box):
        self.Train_RawImg = True
        self.proj =proj
        ##Info from LabelImged
        detection_root = r'G:\DefectDataCenter\ParseData\Detection'
        self.xmlfileDir = r'{}\COT_LabelImged\XML'.format(detection_root)       #存放xml的文件夹
        self.img_dir = r'{}\COT_LabelImged\COT_RAW'.format(detection_root)      #存放图片的文件夹
        #general pytorch info
        self.clsMap = {'1': 'Crack', '2': 'Chipping', '3': 'Particle', '4': 'Discolor', '5': 'Strip', '6': 'Strain',
                       '7': 'others2'}
        #para for Crop
        self.resize2raw_ratio = 2.5  # resize ---> raw 的缩放比例
        self.shrink_w=2  ##分割缺陷时候，收缩缺陷，不至于缺陷位于Img_Crop的边缘
        self.train_percent = 0.8
        self.Croped_img =Croped_img_Size  #[h,w]   [424, 2688]  [848, 896]
        self.xt_r,self.yt_u ,self.xt_l,self.yt_d= Vaild_Box
        #开辟结果存放Dir
        self.folder_initial(detection_root, proj)

    def folder_initial(self,detection_root, proj):
        #dst dir
        self.box_files = ['train.txt', 'val.txt']  # 提取xml ，生成原始的带训练数据
        self.saveConfigDir = r'{}\{}\raw_data\Config'.format(detection_root, proj)
        self.saveDefectDir =  r'{}\{}\raw_data\defect'.format(detection_root, proj)
        self.saveCOTCropImgDir =r'{}\{}\raw_data\Img'.format(detection_root, proj)
        self.saveClassIconDir = r'{}\{}\raw_data\ClassesIcon'.format(detection_root, proj)
        self.ini_path = os.path.join(self.saveConfigDir, "{}_Para.ini").format(proj)

        os.makedirs(self.saveConfigDir, exist_ok=True)
        os.makedirs(self.saveDefectDir, exist_ok=True)
        os.makedirs(self.saveCOTCropImgDir, exist_ok=True)
        os.makedirs(self.saveClassIconDir, exist_ok=True)
        for k,v in self.clsMap.items():
            dft_folder = os.path.join( self.saveDefectDir,v)
            os.makedirs(dft_folder,exist_ok=True)
        #输出Classes.txt
        with open(os.path.join(self.saveConfigDir, "Classes.txt"), 'w') as f:
            for k, v in self.clsMap.items():
                f.write(v + "\n")
        #输出ini文件
        self.gen_MILDataSetParaI_ini()

    def get_train_val_xml(self):
        total_xml = os.listdir(self.xmlfileDir)
        total_xml = [xml for xml in total_xml if xml.endswith('xml')]
        random.shuffle(total_xml)
        train_num = int(len(total_xml) * self.train_percent)
        train_xml = total_xml[:train_num]
        val_xml = total_xml[train_num:]
        return train_xml,val_xml

    def resize_convert_annotation(self, image_id, list_file):
        xml_src_path = os.path.join(self.xmlfileDir,image_id)
        tree = ET.parse(xml_src_path)
        root = tree.getroot()
        # 读取各个box数据
        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # difficult==1（是否在边界）和不在classes中的class不记录
            if cls not in list(self.clsMap.keys()) or int(difficult) == 1:
                continue
            cls_id = list(self.clsMap.keys()).index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(int(a*self.resize2raw_ratio)) for a in b]) + ',' + str(cls_id))

    def get_box_resize(self):
        ##读取xml文件，并划分训练、测试list
        train_xmls,val_xmls = self.get_train_val_xml()
        #写入train的box
        for xmls,file_n in zip([train_xmls,val_xmls], self.box_files):
            dst_path =os.path.join( self.saveConfigDir,file_n)
            file = open(dst_path, 'w')
            for txml in xmls:
                img_path =  os.path.join(self.img_dir,txml.replace('xml','bmp'))
                file.write(img_path)
                self.resize_convert_annotation(txml, file)
                file.write('\n')
            file.close()
        return

    def get_adaptedCropedImg_wid(self,x1,x2,adapted_Img_len,dft_shift_x,img_w):
        ##若x2-x1>adapted_Img_len(train_img_size)，缺陷过大，进行切割
        #返回adapted_Img 中defect的location
        dft_center = (x1+x2)//2
        adapted_Img_x1 = max(dft_shift_x,dft_center-adapted_Img_len//2)
        adapted_Img_x1 = min(adapted_Img_x1,img_w-adapted_Img_len-dft_shift_x)
        adapted_Img_x2 = adapted_Img_x1 +adapted_Img_len
        # defect 在adapted_Img中的location
        new_dft_x1 = max(x1-adapted_Img_x1,0)
        new_dft_x2 = min(x2-adapted_Img_x1,adapted_Img_len)
        return adapted_Img_x1,adapted_Img_x2,new_dft_x1,new_dft_x2

    def box_in_img(self,tmp_box,tmp_CropedImg):
        x_in = tmp_CropedImg[0]<tmp_box[0] and tmp_box[2]<tmp_CropedImg[2]
        y_in = tmp_CropedImg[1] < tmp_box[1] and tmp_box[3] < tmp_CropedImg[3]
        return x_in and y_in

    def seg_box_img_way_2(self,img_box_list,t_box):
        #   解析train_box/val_box
        #   COT ------>
        #   1、提取所有boxes
        #   2、遍历所有boxes，遍历过程中判断该序号是否在已经遍历的set中，并设置已经遍历缺陷的序号集
        #   3、按缺陷裁剪出包含缺陷的CropedImg，存储adapted_box
        #   4、将该缺陷序号加入已遍历序号集，遍历缺陷是否是否存在被该CropedImg包含的defect，若有，加入序号集，存储adapedted_box
        #   5、直到所有缺陷遍历完
        ##3.1、设置包含缺陷的CropedImg
        img_p = img_box_list[0]
        img =  cv2.imdecode(np.fromfile(img_p, dtype=np.uint8),-1)
        img_n = os.path.basename(img_p)
        img_h,img_w =img.shape[0],img.shape[1]
        dft_shift_x = 10 #defect相对CropedImg左上角的x向距离
        dft_shift_y = 10 #defect相对CropedImg左上角的y向距离
        boxes = [[int(x) for x in box.strip().split(',')] for box in img_box_list[1:]]
        CropedImg_list = []     #切割完的小图的location
        CropedImg_boxes = []    #切割完的小图所包含的boxes
        while boxes:
            box = boxes.pop()
            x1,y1,x2,y2,cls = box
            #根据box选取自定义的CropedImg
            if y2<self.yt_u or y1>self.yt_d:
                #位于边框上下横向区域,横向切割出图片
                adapted_Img_x1,adapted_Img_x2,new_dft_x1,new_dft_x2 = self.get_adaptedCropedImg_wid(x1,x2,self.Croped_img[1],dft_shift_x,img_w)
                adapted_Img_y1,adapted_Img_y2,new_dft_y1,new_dft_y2 = self.get_adaptedCropedImg_wid(y1, y2, self.Croped_img[0], dft_shift_y, img_h)
            elif x2<self.xt_r or x1>self.xt_l:
                #位于边框左右竖向区域,竖向切割出图片
                adapted_Img_x1,adapted_Img_x2,new_dft_x1,new_dft_x2 = self.get_adaptedCropedImg_wid(x1,x2,self.Croped_img[0],dft_shift_x,img_w)
                adapted_Img_y1,adapted_Img_y2,new_dft_y1,new_dft_y2 = self.get_adaptedCropedImg_wid(y1, y2, self.Croped_img[1], dft_shift_y, img_h)
            else:
                #位于边框内部区域，按照2*2切割图像
                continue
            tmp_CropedImg = [adapted_Img_x1,adapted_Img_y1,adapted_Img_x2,adapted_Img_y2]
            CropedImg_list.append(tmp_CropedImg)
            adapted_box =  [[new_dft_x1,new_dft_y1,new_dft_x2,new_dft_y2,cls]]
            CropedImg_boxes.append(adapted_box)
            tmp_boxes = []
            if boxes:
                for tmp_box in boxes:
                    tx1, ty1, tx2, ty2,tcls = tmp_box
                    if self.box_in_img(tmp_box,tmp_CropedImg):
                        tmp_boxes.append(tmp_box)
                        tmp_adapted_box = [tx1 - adapted_Img_x1, ty1 - adapted_Img_y1, tx2 - adapted_Img_x1, ty2 - adapted_Img_y1,tcls]
                        adapted_box.append(tmp_adapted_box)
            #去除新增的boxes
            if tmp_boxes:
                [boxes.remove(x) for x in tmp_boxes]
        #对img 进行切割，CropedImg--CropedImg_boxes,存放入txt文件
        #每迭代一次，即为img_box_info的一行
        for croped_i,(img_loc,dft_boxes) in enumerate(zip(CropedImg_list,CropedImg_boxes)):
            #img_loc不为空，CropedImg处于边框区域
            #缺陷位于内部的区域暂不考虑
            if img_loc:
                img_x1,img_y1,img_x2,img_y2  = img_loc
                if img_x2-img_x1>img_y2-img_y1:
                    #横向CropedImg
                    img_i = img[img_y1:img_y2,img_x1:img_x2]
                    img_i_n = str(croped_i)+"_"+img_n
                    img_i_p = os.path.join(self.saveCOTCropImgDir,img_i_n)
                    cv2.imwrite(img_i_p,img_i)
                    img_box_info = img_i_p
                    for bi,tmp_dft_box in enumerate(dft_boxes):
                        img_box_info += " "+",".join([str(x) for x in tmp_dft_box])
                        ##保存切割的dft
                        dft_x1,dft_y1,dft_x2,dft_y2,cls = tmp_dft_box
                        dft = img_i[dft_y1:dft_y2,dft_x1:dft_x2]
                        dft_p = os.path.join(self.saveDefectDir,self.clsMap[str(cls+1)],"croped_"+str(croped_i)+"_box_"+str(bi)+img_i_n)   ##saveDefectDir  saveCropedDefectDir
                        # print("tmp_dft_box: ",tmp_dft_box)
                        # print("dft_p: ", dft_p)
                        cv2.imwrite(dft_p,dft)
                        # 存入Icon
                        ClassIcon_p = os.path.join(self.saveClassIconDir, self.clsMap[str(cls+1)] + ".bmp")
                        if not os.path.exists(ClassIcon_p):
                            cv2.imwrite(ClassIcon_p, dft)
                    t_box.write(img_box_info+"\n")
                else:
                    #竖向CropedImg，对Img进行顺时针旋转270°，box重新定位
                    #左上角点：y' = w-x2 ;x' = y1;
                    img_i = img[img_y1:img_y2,img_x1:img_x2]
                    img_i_rat = cv2.rotate(img_i, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_i_n = str(croped_i)+"Rat_"+img_n
                    img_i_p = os.path.join(self.saveCOTCropImgDir,img_i_n)
                    cv2.imwrite(img_i_p,img_i_rat)
                    img_box_info = img_i_p
                    for bi,tmp_dft_box in enumerate(dft_boxes):
                        #h,w 转换
                        box_h = tmp_dft_box[2]-tmp_dft_box[0]
                        box_w = tmp_dft_box[3] - tmp_dft_box[1]
                        left_up_x = tmp_dft_box[1]   #x' = y1;
                        left_up_y = self.Croped_img[0]-tmp_dft_box[2]  #y' = w-x2
                        dft_box_rat = [left_up_x,left_up_y,left_up_x+box_w,left_up_y+box_h,tmp_dft_box[-1]]
                        img_box_info += " "+",".join([str(x) for x in dft_box_rat])
                        ##保存切割的dft
                        dft_x1,dft_y1,dft_x2,dft_y2,cls = dft_box_rat
                        dft = img_i_rat[dft_y1:dft_y2,dft_x1:dft_x2]
                        dft_p = os.path.join(self.saveDefectDir,self.clsMap[str(cls+1)],"croped_"+str(croped_i)+"Rat__box_"+str(bi)+img_i_n)   ##saveDefectDir  saveCropedDefectDir
                        # print("tmp_dft_box: ",dft_box_rat)
                        # print("dft_p: ", dft_p)
                        cv2.imwrite(dft_p,dft)
                        # 存入Icon
                        ClassIcon_p = os.path.join(self.saveClassIconDir, self.clsMap[str(cls+1)] + ".bmp")
                        if not os.path.exists(ClassIcon_p):
                            cv2.imwrite(ClassIcon_p, dft)
                    t_box.write(img_box_info+"\n")
        return

    def merge2txt(self,src_file_path1,src_file_path2,dst_file_path):
        with open(dst_file_path,'w') as dst_file:
            with open(src_file_path1,'r') as f1:
                f1_lines =f1.readlines()
                for f1_line in f1_lines:
                    dst_file.write(f1_line)
            with open(src_file_path2,'r') as f2:
                f2_lines =f2.readlines()
                for f2_line in f2_lines:
                    dst_file.write(f2_line)
        return

    def gen_MILDataSetParaI_ini(self,):
        with open(self.ini_path,'w') as f:
            f.write("[{}]\n".format(self.proj))
            f.write("ClassesPath={}\Classes.txt\n".format(self.saveConfigDir))
            IconDir = self.saveConfigDir.replace("Config",'')
            f.write("IconDir={}ClassesIcon\ \n".format(IconDir))
            # f.write("TrainDataInfoPath={}\ImgBoxes_MIL_train.txt\n".format(self.saveConfigDir))
            f.write("TrainDataInfoPath={}\ImgBoxes_train.txt\n".format(self.saveConfigDir))
            f.write("ValDataInfoPath={}\ImgBoxes_val.txt\n".format(self.saveConfigDir))

            MILDir = IconDir.replace('raw_data','MIL_Data')
            f.write("WorkingDataDir={} \n".format(MILDir))
            f.write("PreparedDataDir={}PreparedData\n".format(MILDir))
            f.write("ImageSizeX={}\n".format(self.Croped_img[1]))
            f.write("ImageSizeY={}\n".format(self.Croped_img[0]))
            f.write("AugFreq=0\n")
            f.write("TestDataRatio=10\n")
        return

    def crop_COT(self):
        #step1:生成原始的'val.txt','train.txt' 【针对小图可以直接使用】，可以直接使用的xml文件，#若xml文件来自于尺寸合适的Img，则'val.txt','train.txt'可直接使用
        self.get_box_resize()
        #step2:由于labelImg 生成的xml文件使用的是大图，因此要根据缺陷位置进行裁剪
        for box_file in self.box_files:
            path = os.path.join(self.saveConfigDir,box_file)
            with open(path, 'r') as f:
                img_boxes = f.readlines()
            new_box_path = os.path.join(self.saveConfigDir,"ImgBoxes_"+box_file)
            with open(new_box_path,'w') as t_box:
                for img_box in img_boxes:
                    img_box_list= img_box.split(' ')
                    #  自适应裁剪得到 尺寸 [848, 896]  #[h,w]的训练图片，以及defect
                    self.seg_box_img_way_2(img_box_list,t_box)
        #拼接ImgBoxes_train、ImgBoxes_val
        src_file_path1 = os.path.join(self.saveConfigDir, "ImgBoxes_train.txt")
        src_file_path2 = os.path.join(self.saveConfigDir, "ImgBoxes_val.txt")
        dst_file_path =  os.path.join(self.saveConfigDir, "ImgBoxes_MIL_train.txt")
        self.merge2txt(src_file_path1, src_file_path2, dst_file_path)
        #生成COT_Resize_Para.ini 文件
        # self.gen_MILDataSetParaI_ini()

if __name__ == '__main__':
    # project = 'VOC'
    # test = XMLExtract(project)
    # test.get_box()

    # project = 'COT_Resize'
    # Croped_img_Size = [224, 1120]  # [h,w]   [424, 2688]  [848, 896]
    # Vaild_Box = [120,120,4360,4920]#x1,y1,x2,y2

    project = 'COT_Raw'
    Croped_img_Size = [424, 2688]  # [h,w]   [424, 2688]  [848, 896]
    Vaild_Box = [300,300,10900,12300]#x1,y1,x2,y2
    test = ExtractResizeCOTXML2Raw(project,Croped_img_Size,Vaild_Box)
    test.crop_COT()



