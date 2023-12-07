import os
import xml.etree.ElementTree as ET

def resize_convert_annotation(xml_src_path, list_file,classes,img_path=''):
    tree = ET.parse(xml_src_path)
    root = tree.getroot()
    # 读取各个box数据
    if img_path=='':
        for oj in root.iter('annotation'):
            img_path = oj.find('path').text
    img_info = img_path+' '
    for obj in root.iter('object'):

        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        print("cls: {}".format(cls))
        if cls not in classes:
            classes.append(cls)
        # difficult==1（是否在边界）和不在classes中的class不记录
        if int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        box_str = xmlbox.find('xmin').text+','+xmlbox.find('ymin').text+','+\
                  xmlbox.find('xmax').text+','+ xmlbox.find('ymax').text+','+cls+' '
        img_info +=box_str
    img_info = img_info.rstrip()
    list_file.write(img_info)
    return

def xml2txt(xml_src,src_img_dir,dst_xml_path,dst_cls_path):

    print("xml_src:{}".format(xml_src))
    print("src_img_dir:{}".format(src_img_dir))
    print("dst_xml_path:{}".format(dst_xml_path))
    print("dst_cls_path:{}".format(dst_cls_path))
    ##单独的一项提取xml--->txt 的解析
    files = os.listdir(xml_src)
    xmls = [file for file in files if file.endswith('xml')]
    file = open(dst_xml_path, 'w')
    classes = []
    for txml in xmls:
        img_path = os.path.join(src_img_dir, txml.replace('xml', 'bmp'))
        txml_path = os.path.join(xml_src,txml)
        resize_convert_annotation(txml_path, file,classes,img_path=img_path)
        file.write('\n')
    file.close()
    #写入classes
    with open(dst_cls_path,'w') as f:
        for cls in classes:
            f.write(cls+"\n")
    return

if __name__ == '__main__':

    xml_src = r'G:\DefectDataCenter\ParseData\Detection\COT_LabelImged\XML'
    src_img_dir = r'G:\DefectDataCenter\ParseData\Detection\COT_LabelImged\COT_RAW'
    dst_xml_path = r'G:\DefectDataCenter\ParseData\Detection\COT_LabelImged\{}'.format('cot_xml2txt.txt')
    dst_cls_path = r'G:\DefectDataCenter\ParseData\Detection\COT_LabelImged\{}'.format('cot_classes.txt')
    xml2txt(xml_src,src_img_dir,dst_xml_path,dst_cls_path)

