#-----------------------------------------------------------------------#
#   使用模型进行预测，并将预测结果以图片形式保存在路径：'data/{project}/rst_img'
#-----------------------------------------------------------------------#
import time
import cv2
import os
from yolo import YOLO
from utils.config import Config
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    project = "lslm"  ##LMK
    cfg =Config(project)
    yolo = YOLO(cfg)
    img_type = 'jpg'
    img_dir = r'G:\DefectDataCenter\General_Data\{}\val'.format(project)
    img_path = [x for x in os.listdir(img_dir)if x.endswith(img_type)]

    t1 = time.time()
    for img_id,img_name in enumerate(img_path):
        img_id += 1
        img = os.path.join(img_dir, img_name)
        r_image = yolo.detect_cv_image(img)
        image_name = str(img_id) + "_img.bmp"
        dst_path = os.path.join(cfg.dst_dir, image_name)
        cv2.imwrite(dst_path, r_image)

    fps = (len(img_path) / (time.time() - t1))
    print("fps: ", fps)
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
