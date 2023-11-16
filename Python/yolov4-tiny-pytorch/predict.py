#-----------------------------------------------------------------------#
#   使用模型进行预测，并将预测结果以图片形式保存在路径：'data/{project}/rst_img'
#-----------------------------------------------------------------------#
import time
import cv2
import os
from yolo import YOLO
from utils.config import Config


def get_predict_path_way1(project):
    #从图像文件夹中提取出Img_List
    img_type = 'bmp'
    img_dir = r'I:\MIL_AI\Python\yolov4-tiny-pytorch\data\{}\Img'.format(project)
    img_ns = [x for x in os.listdir(img_dir) if x.endswith(img_type)]
    img_path = [os.path.join(img_dir, img_n) for img_n in img_ns]
    return img_path

def get_predict_path_way2(project):
    #从ImgBoxes_val.txt提取出img_list
    # img_txt_info = r'G:\DefectDataCenter\ParseData\Detection\{}\raw_data\Config\Val.txt'.format(project)
    img_txt_info = r'G:\DefectDataCenter\ParseData\Detection\{}\raw_data\Config\ImgBoxes_val.txt'.format(project)

    img_paths = []
    with open(img_txt_info,'r') as f:
        fls = f.readlines()
        for fl in fls:
            img_p = fl.split(" ")[0]
            img_paths.append(img_p)

    return img_paths


def Predict():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    project = "COT_Raw"  ##LMK lslm COT_Raw  COT_Resize  HW  COT_Raw
    cfg = Config(project)
    yolo = YOLO(cfg)
    # img_ps = get_predict_path_way1(project)
    img_ps = get_predict_path_way2(project)

    t1 = time.time()
    results = []
    for img_id, img_p in enumerate(img_ps):
        img_id += 1
        r_image,result = yolo.detect_cv_image(img_p)
        image_name = os.path.basename(img_p)
        dst_path = os.path.join(cfg.predict_dst, image_name)
        cv2.imwrite(dst_path, r_image)
        results.append(result)
    fps = (len(img_ps) / (time.time() - t1))
    print("fps: ", fps)
    ##将results写入 result file 中
    dst_path =  r'G:\DefectDataCenter\ParseData\Detection\{}\raw_data\ImgBoxes_val_pd_result.txt'.format(project)
    with open(dst_path,'w') as f:
        for i in range(50,len(results)):
            img_i_rst = results[i]
            if img_i_rst !=" ":
                box_info = ""
                for j in range(len(img_i_rst[0])):
                    box_i = img_i_rst[0][j]
                    box_info += ",".join([str(int(x)) for x in box_i[:4]])+","+str(int(box_i[6]))+" "
                n_box_info = box_info.rstrip()+"\n"
                f.write(n_box_info)
            else:
                f.write("0,0,0,0,0\n")  #未检测出box，以0代替
    return results



def ValModel_AP_50():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    project = "DSW"  ##LMK lslm COT_Raw  COT_Resize ;DSW_random ;DSW COT_Raw HW
    cfg = Config(project)
    yolo = YOLO(cfg)
    img_ps =cfg.val_lines

    # img_ps = cfg.train_lines
    t1 = time.time()
    dst_img_rst = os.path.join(cfg.dst_root,'img_rst')
    dst_pd_txt = os.path.join(cfg.dst_root,'ImgBoxes_val_pd_result.txt')
    os.makedirs(dst_img_rst,exist_ok=True)
    results = []
    for img_id, img_info in enumerate(img_ps):
        img_id += 1
        img_p = img_info.split(" ")[0]
        r_image,result = yolo.detect_cv_image(img_p)
        image_name = os.path.basename(img_p)
        dst_path = os.path.join(dst_img_rst, image_name)
        cv2.imwrite(dst_path, r_image)
        results.append(result)
    fps = (len(img_ps) / (time.time() - t1))
    print("fps: ", fps)
    ##将results写入 result file 中
    with open(dst_pd_txt,'w') as f:
        for i in range(len(results)):
            img_i_rst = results[i]
            if img_i_rst !=" ":
                box_info = ""
                for j in range(len(img_i_rst[0])):
                    box_i = img_i_rst[0][j]
                    box_info += ",".join([str(int(x)) for x in box_i[:4]])+","+str(int(box_i[6]))+" "
                n_box_info = box_info.rstrip()+"\n"
                f.write(n_box_info)
            else:
                f.write("0,0,0,0,0\n")  #未检测出box，以0代替
    return results


if __name__ == "__main__":
    Predict()
    # ValModel_AP_50()
