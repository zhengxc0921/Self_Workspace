from image_classifier.net_operate import *


#
# 完成时间：2020-11-11
# 功能：兼容双模型，兼容多分类
class Classifier:
    def __init__(self,img_argu):
        self.img_argu = img_argu
        self.net_operate = NetOperate(img_argu)

    def train(self):
        self.net_operate.model_train()

    def test(self):
        # 模型测试:
        self.net_operate.model_load(self.img_argu.best_model_path)
        self.net_operate.model_evaluate(self.img_argu.val_loader)

    def update(self):
        # 模型更新
        self.net_operate.model_load(self.img_argu.best_model_path)
        self.net_operate.model_train()

    def predict_initial(self):
        self.dst_dir = os.path.join(self.img_argu.dst_root,self.img_argu.damage_fold)
        for k, v in self.img_argu.name2label.items():
            p_dir = os.path.join(self.dst_dir, str(k))
            os.makedirs(p_dir, exist_ok=True)

    def predict(self,src_dir):
        # 模型预测:预测无标签数据的类别
        self.predict_initial()
        self.net_operate.model_load(self.img_argu.best_model_path)
        self.net_operate.model_predict(self.dst_dir,src_dir)

    def clac_cam(self,dst_dir):
        # 模型预测:预测无标签数据的类别
        self.net_operate.model_load()
        self.net_operate.model_cam(dst_dir,self.img_argu.pred_loader)

def Run(calc_device):
    img_argu = Argu(calc_device)
    test = Classifier(img_argu)
    t1 = time.time()
    test.train()
    test.test()
    # t2 = time.time()
    # test.update()
    # test.test()
    # 模型预测，数据表制作，更新
    src_dir = r'G:\DefectDataCenter\ParseData\Classifier\SXX_GrayWave\Original_Gray3\91'
    test.predict(src_dir)
    t2 = time.time()
    print("耗时：{} s".format(t2 - t1))

if __name__ == '__main__':
    calc_device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    Run(calc_device)
