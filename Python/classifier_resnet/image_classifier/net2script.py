import torch
import os
from image_classifier.argu import Argu
import numpy as np
import cv2,shutil

class ToScript:

    def __init__(self,calc_device):
        self.calc_device = calc_device
        self.argu = Argu(calc_device)
        self.model = self.argu.model.to(calc_device)
        self.model.load_state_dict(torch.load(self.argu.best_model_path))
        self.onnx_model_dst = self.argu.onnx_m_dst
        self.example_img_initial()

    def cv_read_(self,image_path):
        # opencv2中的双线性插值法：INTER_LINEAR和
        # transform.resize()的BILINEAR插值的差异
        img_cv_gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        img_cv_BGR = cv2.cvtColor(img_cv_gray,cv2.COLOR_GRAY2BGR)/ 255.0
        img_cv_resize = cv2.resize(img_cv_BGR, self.argu.input_size, interpolation=cv2.INTER_LINEAR)
        img_cv_resize_np_f64 = np.transpose(img_cv_resize, [2, 0, 1])
        img_cv_resize_np = np.array(img_cv_resize_np_f64, dtype='float32')
        example_low_d = torch.tensor(img_cv_resize_np)
        example = torch.unsqueeze(example_low_d, 0).to(self.calc_device)
        return example

    def toscript(self):
        # 读取ones结果与C++一致
        # new_img = np.ones((1, 3, 224, 224))
        # example = torch.tensor(new_img, dtype=torch.float32)/255.0
        self.model.eval()  # 指定当前模型是在训练（#不使用BatchNormalization()和Dropout()）
        print(self.model(self.ep_img))
        traced_script_module = torch.jit.trace(self.model,self.ep_img)
        traced_script_module.save(self.argu.pt_m_dst)

    def example_img_initial(self):
        self.ep_img_p = os.path.join(self.argu.src_dir,self.argu.label2class[0])
        self.ep_im_ns= os.listdir(self.ep_img_p)
        img_path = os.path.join(self.ep_img_p,self.ep_im_ns[0])
        self.ep_img = self.cv_read_(img_path)
        dst_img_p = os.path.join(self.argu.model_dst_dir,self.ep_im_ns[0])
        shutil.copy(img_path,dst_img_p)


    def to_onnx(self):
        import onnx ,onnxruntime
        # withe_img = np.ones((224,224,3))
        # cv2.imwrite('withe_img.bmp',withe_img)
        # new_img = np.ones((1, 3, 224, 224))
        # images_cv = torch.tensor(new_img, dtype=torch.float32)/255.0
        # self.ep_img = images_cv
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        self.model.eval()
        test_in = self.model(self.ep_img)
        print("test_in: ", test_in)
        input_layer_names = ["images"]
        output_layer_names = ["outputs"]
        torch.onnx.export(self.model, self.ep_img,
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
            test_out = sess.run(["outputs"], {"images": np.array(self.ep_img)})
            print("test_out: ", test_out)
        except:
            print("this image have no defects!")
        return



if __name__ == '__main__':
    image_path= r'G:\DefectDataCenter\ParseData\Classifier\SXX_GrayWave\DataSet\prepare\Images\10\1_2_3_4_8_3604_7822_1(patch).ac405cb244_Prp_0.mim'
    img_cv_gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

    calc_device = 'cpu'
    test1 = ToScript(calc_device)
    test1.toscript()
    test1.to_onnx()
