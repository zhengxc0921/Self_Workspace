import numpy as np
import cv2,os

src_dir = 'I:/MIL_Detection_Dataset/VOC/'
dst_dir = src_dir+ "ODNetResult_show/"
os.makedirs(dst_dir,exist_ok=True)

# cns = ['LargeKnot','SmallKnot']
# cns = ['bolt','nut']
# cns = ['0','1']
cns = ['aeroplane',
'bicycle',
'bird',
'boat',
'bottle',
'bus',
'car',
'cat',
'chair',
'cow',
'diningtable',
'dog',
'horse',
'motorbike',
'person',
'pottedplant',
'sheep',
'sofa',
'train',
'tvmonitor']




rect_color_list = [(255,0,0),(0,255,0),(0,0,255)]
cn_c = {}
for i,cn in enumerate(cns):
    cn_c[cn] = rect_color_list[i%3]



def visualize_bbox(img, bbox ,score, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    # y_min ,x_min, y_max ,x_max = bbox

    x_min,y_min, x_max,y_max  = bbox

    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    y_min_n = y_min - int(1.3 * text_height)
    x_max_n = x_max + text_width
    cv2.rectangle(img, (int(x_min), int(y_min_n)), (int(x_max), int(y_min)), color=color, thickness=-1)
    cv2.putText(
        img,
        text='{} {:.2f}'.format(class_name, score),
        org=(int(x_min), int(y_min) - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image_path, class_names,top_conf, top_boxes):
    # img_d = np.transpose(img,(1,0,2))
    ##遍历所有图片
    for i in range(len(image_path)):
        img = cv2.imdecode(np.fromfile(image_path[i], dtype=np.uint8), 1)
        img = cv2.resize(img,(512,512))
        cn = class_names[i]
        score = top_conf[i]
        bbox = top_boxes[i]
        ##遍历单张图片中的所有box
        for j in range(len(cn)):


            rect_color = cn_c[str(cn[j])]
            img = visualize_bbox(img, bbox[j], score[j], str(cn[j]),rect_color)
        # cv2.imshow("img",img)
        dst_path = dst_dir + image_path[i].split("/")[-1]
        if "mim" in dst_path:
            dst_path = dst_path.replace("mim","jpg")
        cv2.imwrite(dst_path, img)
    return




# # ---------------------------------------------------------#
# #   设置字体与边框厚度
# # ---------------------------------------------------------#
# font = ImageFont.truetype(font='model_data/simhei.ttf',
#                           size=np.floor(3e-2 * image.shape[1] + 0.5).astype('int32'))
# thickness = int(max((image.shape[0] + image.shape[1]) // np.mean(self.input_shape), 1))

# ---------------------------------------------------------#
#   图像绘制
# ---------------------------------------------------------#
# visualize(image_raw, top_label, top_conf, top_boxes)

def parse_ODNetResult():

    src_path = src_dir + 'ODNetResult.txt'
    with open(src_path,'r') as od_f:
        fl =od_f.readlines()
        imgs_path = []
        vec_boxes = []
        vec_conf = []
        vec_cn = []
        for line in fl:

            boxes = []
            conf = []
            cn =[]
            vecline = line.split(" ")
            #统计单张图片数据
            if len(vecline)>1:

                for i in range(1,len(vecline),6):
                    box = np.zeros(4)
                    box[0] = float(vecline[i])-float(vecline[i + 2])/2
                    box[1] = float(vecline[i + 1]) - float(vecline[i + 3])/2
                    box[2] = float(vecline[i])+float(vecline[i + 2])/2
                    box[3] = float(vecline[i + 1]) + float(vecline[i + 3])/2
                    boxes.append(box)
                    conf.append(float(vecline[i + 4]))
                    cn.append(vecline[i + 5].strip())
            imgs_path.append(vecline[0].strip())
            vec_boxes.append(boxes)
            vec_conf.append(conf)
            vec_cn.append(cn)

    visualize(imgs_path, vec_cn, vec_conf, vec_boxes)
    return


def parse_ODNetInput():

    src_path = src_dir + 'ImgBoxes.txt'
    with open(src_path,'r') as od_f:
        fl =od_f.readlines()
        imgs_path = []
        vec_boxes = []
        vec_conf = []
        vec_cn = []
        for line in fl:

            boxes = []
            conf = []
            cn =[]
            vecline = line.split(" ")
            #统计单张图片数据
            if len(vecline)>1:

                for i in range(1,len(vecline)):
                    vecline_i =  [float(x) for x in vecline[i].split(',')]
                    # box = np.zeros(4)
                    # box[0] = float(vecline_i[i])-float(vecline_i[i + 2])/2
                    # box[1] = float(vecline_i[i + 1]) - float(vecline_i[i + 3])/2
                    # box[2] = float(vecline_i[i])+float(vecline_i[i + 2])/2
                    # box[3] = float(vecline_i[i + 1]) + float(vecline_i[i + 3])/2
                    boxes.append(vecline_i[:4])
                    conf.append(1.0)
                    cn.append(int(vecline_i[4]))
            imgs_path.append(vecline[0].strip())
            vec_boxes.append(boxes)
            vec_conf.append(conf)
            vec_cn.append(cn)

    visualize(imgs_path, vec_cn, vec_conf, vec_boxes)
    return


if __name__ == '__main__':
    # parse_ODNetInput()
    parse_ODNetResult()
