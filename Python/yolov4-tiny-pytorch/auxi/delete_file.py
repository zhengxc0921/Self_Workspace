import os
import shutil
def de_file_dir():
    dir = r'G:\DefectDataCenter\邓军法\imgs_120221026012044'
    paths= os.listdir(dir)
    for p in paths:
        path = os.path.join(dir,p)
        os.remove(path)  # 删除文件
    os.removedirs(dir)  # 删除空文件夹


def file_move(src_dir,dst_dir):
    # src_path = '/home/A_codeTest/train/move1.jpg'
    # dst_path = '/home//A_codeTest/move'
    for file in os.listdir(src_dir):
        if '.' not in file :
            tp_src_dir = os.path.join(src_dir,file)
            tp_dst_dir = os.path.join(dst_dir, file)
            os.makedirs(tp_dst_dir,exist_ok=True)
            file_move(tp_src_dir, tp_dst_dir)
        else:
            src_path = os.path.join(src_dir,file)
            dst_path = os.path.join(dst_dir, file)
            shutil.move(src_path, dst_path)

# os.rmdir(path)  # 删除空文件夹
# shutil.rmtree(path)  # 递归删除文件夹，即：删除非空文件夹
if __name__ == '__main__':
    # for i in range(9,11):
    #     path = r'G:\DefectDataCenter\wangrixuan\wuxi_WSD300\defect\prepare{}'.format(i)
    #     shutil.rmtree(path)
    #
    path = r'G:\classification_data_lib\Train_data'
    shutil.rmtree(path)

    # src_dir = r'G:\classification_data_lib\Train_data\Mg_grain_v2'
    # dst_dir =r'G:\DefectDataCenter\ParseData\Classifier\Mg_grain_v2'
    # file_move(src_dir, dst_dir)
