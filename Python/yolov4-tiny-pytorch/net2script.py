import torch
from PIL import Image
from yolo import YOLOScript

from nets.yolo import YoloBodyS
import numpy as np
import cv2,os


class ToScript:

    def __init__(self,cfg):
        self.model_YOLO =YOLOScript(cfg)
        self.model_dst = cfg.pt_model_dst
        self.onnx_model_dst = cfg.onnx_model_dst
        self.cfg = cfg

        self.img_dir = r'I:\MIL_Detection_Dataset\{}\raw_data\img'.format(project)
        # img_path = os.listdir(img_dir)
        # self.img_dir = cfg.dir_origin_path


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
        # img_pil_resize = np.array(image.resize((416, 352), Image.BICUBIC))/255
        images_cv = torch.from_numpy(np.expand_dims(np.transpose(img_cv_resize,(2,0,1)),0))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        # image_data  = np.expand_dims(np.transpose(self.preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # print("image_data[0,0,0,:416]: ",image_data[0, 0, 0, :416])
        # images = torch.from_numpy(image_data)
        with torch.no_grad():
            ###测试body
            # model_YoloBody_result = self.model_YoloBody(images)
            # print("model_YoloBody_result:",model_YoloBody_result)
            # traced_script_module = torch.jit.script(self.model_YoloBody)
            # traced_script_module.save(self.model_YoloBody_dst)
            # AA_model_YoloBody_dst = torch.jit.load(self.model_YoloBody_dst)
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



    def to_onnx(self):
        import onnx ,onnxruntime
        model_dst = "yolo4_tiny_onnx.onnx"
        input_layer_names = ["images"]
        output_layer_names = ["output"]
        # Export the model
        img = torch.zeros((1, 3) + tuple(self.cfg.input_shape)).type(torch.float32) # (1, 3, 320, 192)
        # torch.onnx.export(self.model_YOLO, img, self.onnx_model_dst, verbose=True)
        # print(f'Starting export with onnx {onnx.__version__}.')
        images_cv =   np.zeros((1, 3) + tuple(self.cfg.input_shape)).astype(np.float32) # (1, 3, 320, 192
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.model_YOLO,
                          img,
                          f=model_dst,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)
        # Checks
        model_onnx = onnx.load(model_dst)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # onnx 模型猜测是
        # 创建一个InferenceSession的实例，并将模型的地址传递给该实例
        sess = onnxruntime.InferenceSession(model_dst)
        # self.input_name = self.get_input_name(sess)
        # self.output_name = self.get_output_name(sess)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))
        test_out = sess.run(["output"], {"images": images_cv})
        print("test_out:{}".format(test_out))



if __name__ == '__main__':
    # calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pj_id = 1
    from utils.config import Config

    project = "lslm_all"  #LMK
    cfg = Config(project)
    test1 = ToScript(cfg)
    test1.to_onnx()
