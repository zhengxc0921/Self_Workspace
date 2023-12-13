from run_func.train import yolo4_train
from run_func.predict import Predict
from auxi.COT_extract import ExtractResizeCOTXML2Raw

from Utils.config import Config
from run_func.net2script import ToScript
def Start_train():
    print("Start Train")
    yolo4_train()
    print("Finish Train")

def Start_predict():
    print("Start Predict")
    Predict()
    print("Finish Predict")

def Start_ParseXML_Resize():
    #打标签的时候，由于原图过大，对原图进行了缩放了resize2raw_ratio，
    #而训练是在原图上截取Crop_h*Crop_w的图片上进行训练的。
    #boundary_width防止截取的图片不靠近边框 300的范围内
    #生成转换后的“ImgBoxes_train.txt”，“ImgBoxes_val.txt”
    print(" Start_ParseXML_Resize")
    project = 'COT_Raw'
    test = ExtractResizeCOTXML2Raw(project)
    resize2raw_ratio, Crop_h, Crop_w, boundary_width = 2.5, 424, 2688, 300
    test.parase_xml2txt_resize(resize2raw_ratio,Crop_h,Crop_w,boundary_width)
    print("Finish ParseXML")

def Start_ParseXML_Direct():
    #打标签的图片和训练图片一致的时候：直接解析xml文件
    #生成“ImgBoxes_train.txt”，“ImgBoxes_val.txt”
    print("Start ParseXML")
    project = 'COT_Raw'
    test = ExtractResizeCOTXML2Raw(project)
    test.parse_xml2txt_direct()
    print("Finish ParseXML")

def Check_TrainDataSet():
    from Utils.config import Config
    from Utils.dataloader import  YoloDataset,yolo_dataset_collate
    project = "DSW_random"  ## LMK, HW,VOC,DSW_random ,COT_Raw ;COT_Raw ; DSW
    cfg = Config(project)
    size_input = [cfg.ImageSizeY, cfg.ImageSizeX]
    train_ls = cfg.train_lines
    ##创建数据集，General
    vis_num = 10
    train_set = YoloDataset(train_ls,size_input)
    train_set.visualize_data(vis_num)


def TOO_ONNX():
    project = "COT_Raw"  #LMK  ## LMK, HW,VOC,DSW_random ,COT_Raw ;COT_Raw ; DSW
    cfg = Config(project)
    test1 = ToScript(cfg)
    # test1.onnx_vison()
    print("test1.onnx_vison()")
    test1.to_onnx()
    print("test1.to_onnx()")

if __name__ == '__main__':
    # Start_ParseXML_Direct()
    # Start_ParseXML_Resize()
    # Check_TrainDataSet()
    # Start_train()
    Start_predict()
    # TOO_ONNX()
    print("Start_predict：Hello World")