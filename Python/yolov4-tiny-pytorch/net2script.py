import torch
from yolo import YOLOScript
import numpy as np
import cv2,os


class ToScript:

    def __init__(self,cfg):
        self.model_YOLO =YOLOScript(cfg)
        self.model_dst = cfg.pth_dst
        self.onnx_model_dst = cfg.onnx_dst
        self.cfg = cfg
        self.img_dir = r'G:\DefectDataCenter\ParseData\Detection\{}\raw_data\TImg'.format(project)

    def toscript(self):
        self.img_names = os.listdir(self.img_dir)
        self.img_path = os.path.join(self.img_dir,self.img_names[10])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        # image       = self.cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_raw = np.array(image, dtype='float32')/255.0
        img_cv_resize = cv2.resize(img_raw,(cfg.input_shape[1],cfg.input_shape[0]),cv2.INTER_CUBIC)
        images_cv = torch.from_numpy(np.expand_dims(np.transpose(img_cv_resize,(2,0,1)),0))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#

        with torch.no_grad():
            ###测试yolo
            model_images_cv_result = self.model_YOLO(images_cv)
            print("model_images_cv_result.size():", model_images_cv_result.size())
            print("model_images_cv_result:",model_images_cv_result)
            # model_pil_result = self.model_YOLO(images)
            # print("model_pil_result:",model_pil_result)
            traced_script_module = torch.jit.script(self.model_YOLO)
            traced_script_module.save(self.model_dst)
            AA_model_dst = torch.jit.load(self.model_dst)
            print("AA_model_dst: ",AA_model_dst(images_cv))
            self.to_onnx(images_cv)
            # print("aa:")
            # print("aaa----AA(example):", aaa -AA(example))
        return

    def pre_process_img(self,num=1):
        self.img_names = os.listdir(self.img_dir)
        self.img_path = os.path.join(self.img_dir,self.img_names[num])
        print("img_path:",self.img_path)
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        # image       = self.cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_raw = np.array(image, dtype='float32') / 255.0
        img_cv_resize = cv2.resize(img_raw, (cfg.input_shape[1], cfg.input_shape[0]), cv2.INTER_CUBIC)
        images_cv = torch.from_numpy(np.expand_dims(np.transpose(img_cv_resize, (2, 0, 1)), 0))
        return images_cv,image.shape[:2]

    def to_onnx(self):
        import onnx ,onnxruntime
        images_cv ,image_shape= self.pre_process_img(0) ##177,4   ##1
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#

        test_in = self.model_YOLO(images_cv)
        print("test_in: ", test_in)
        input_layer_names = ["images"]
        output_layer_names = ["outputs"]

        torch.onnx.export(self.model_YOLO,
                          images_cv,
                          f=self.onnx_model_dst,
                              verbose=False,
                          opset_version=11,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)
        # # # # # Checks
        model_onnx = onnx.load(self.onnx_model_dst)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # onnx 模型猜测是
        # 创建一个InferenceSession的实例，并将模型的地址传递给该实例
        try:
            sess = onnxruntime.InferenceSession(self.onnx_model_dst)
            test_out = sess.run(["outputs"], {"images": np.array(images_cv)})
            print("test_out: ", test_out)
        except:
            print("this image have no defects!")
        return


    def use_onnx(self):
        import  onnxruntime
        n = len(os.listdir(self.img_dir))
        for i in range(n):
            images_cv ,image_shape= self.pre_process_img(i) ##177,4
            #---------------------------------------------------------#
            #   添加上batch_size维度
            #---------------------------------------------------------#
            try:
                sess = onnxruntime.InferenceSession(self.onnx_model_dst)
                test_out = sess.run(["outputs"], {"images": np.array(images_cv)})
                print("test_out: ", test_out)
            except:
                print("this image have no defects!")
        return



if __name__ == '__main__':
    # calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pj_id = 1
    from utils.config import Config
    project = "DSW"  #LMK
    cfg = Config(project)
    test1 = ToScript(cfg)
    test1.to_onnx()
    test1.use_onnx()