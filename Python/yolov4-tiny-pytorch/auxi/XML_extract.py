import os
import random
import xml.etree.ElementTree as ET

# 大流程的Step1
# Step1：打标签，制作xml文件
# Step2：解析xml文件，读取源图路径，box数据，写入txt文件
# Step3：对txt文件划分测试，训练数据集
# xmlfilepath = r'../input/VOC/Annotations'
# saveBasePath = r"../input/VOC/ImageSets/"
class XMLExtract:
    def __init__(self, project):
        self.project = project
        self.xmlfilepath = '../input/' + self.project + '/Annotations'
        self.saveBasePath = '../input/' + self.project + '/ImageSets'
        self.classes_path = '../input/' + self.project + '/config/classes.txt'
        self.img_dir = r'\input\%s\Img'%(self.project)
        self.train_percent = 0.8
        self.get_imgsets()

    def write_txt(self, xml_list, xml_type):
        file_path = os.path.join(self.saveBasePath, xml_type + '.txt')
        fwrite_txt = open(file_path, 'w')
        for xml in xml_list:
            name = xml[:-4] + '\n'
            fwrite_txt.write(name)
        fwrite_txt.close()

    def get_imgsets(self):
        total_xml = os.listdir(self.xmlfilepath)
        random.shuffle(total_xml)
        train_num = int(len(total_xml) * self.train_percent)
        train_xml = total_xml[:train_num]
        val_xml = total_xml[train_num:]
        self.write_txt(train_xml, 'train')
        self.write_txt(val_xml, 'val')

    def read_class(self):
        with open(self.classes_path, 'r') as f:
            classes_tmp = f.readlines()
        self.classes = [str.strip() for str in classes_tmp]

    def convert_annotation(self, image_id, list_file):
        xml_src_path = self.xmlfilepath + r'/%s.xml' % (image_id)
        tree = ET.parse(xml_src_path)
        root = tree.getroot()
        # 读取各个box数据
        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # difficult==1（是否在边界）和不在classes中的class不记录
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    def get_box(self):
        self.read_class()
        for image_set in ['train', 'val']:
            src_path = self.saveBasePath + '/' + image_set + '.txt'
            image_ids = open(src_path).read().split('\n')  # 临时测试
            dst_path = self.saveBasePath + '/' + image_set + '_box.txt'
            list_file = open(dst_path, 'w')
            for image_id in image_ids[:-1]:  # 临时测试
                img_src_path = os.path.dirname(os.getcwd()) + self.img_dir + r'\%s.jpg' % (image_id)
                list_file.write(img_src_path)
                self.convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()


if __name__ == '__main__':
    project = 'VOC'
    test = XMLExtract(project)
    test.get_box()

