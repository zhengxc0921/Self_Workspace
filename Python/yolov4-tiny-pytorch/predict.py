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
    project = "COT_Raw"  ##LMK lslm COT  COT_Resize
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
        dst_path = os.path.join(cfg.dst_dir, image_name)
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


def Predictt():
    print("fps: ")

if __name__ == "__main__":
    Predict()
    # elif mode == "video":
    #     capture = cv2.VideoCapture(video_path)
    #     if video_save_path!="":
    #         fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    #         size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #         out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    #
    #     ref, frame = capture.read()
    #     if not ref:
    #         raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    #
    #     fps = 0.0
    #     while(True):
    #         t1 = time.time()
    #         # 读取某一帧
    #         ref, frame = capture.read()
    #         if not ref:
    #             break
    #         # 格式转变，BGRtoRGB
    #         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #         # 转变成Image
    #         frame = Image.fromarray(np.uint8(frame))
    #         # 进行检测
    #         frame = np.array(yolo.detect_image(frame))
    #         # RGBtoBGR满足opencv显示格式
    #         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #
    #         fps  = ( fps + (1./(time.time()-t1)) ) / 2
    #         print("fps= %.2f"%(fps))
    #         frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    #         cv2.imshow("video",frame)
    #         c= cv2.waitKey(1) & 0xff
    #         if video_save_path!="":
    #             out.write(frame)
    #
    #         if c==27:
    #             capture.release()
    #             break
    #
    #     print("Video Detection Done!")
    #     capture.release()
    #     if video_save_path!="":
    #         print("Save processed video to the path :" + video_save_path)
    #         out.release()
    #     cv2.destroyAllWindows()
    #
    # elif mode == "fps":
    #     img = Image.open('img/street.jpg')
    #     tact_time = yolo.get_FPS(img, test_interval)
    #     print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    #
    # elif mode == "dir_predict":
    #     import os
    #     from tqdm import tqdm
    #
    #     img_names = os.listdir(dir_origin_path)
    #     for img_name in tqdm(img_names):
    #         if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             image_path  = os.path.join(dir_origin_path, img_name)
    #             image       = Image.open(image_path)
    #             r_image     = yolo.detect_image(image)
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             r_image.save(os.path.join(dir_save_path, img_name))
    #
    # else:
    #     raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
