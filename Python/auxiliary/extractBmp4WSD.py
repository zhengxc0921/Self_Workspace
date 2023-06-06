import os,shutil

def extract_bmp4_WSD(src_dir,dst_dir):
    # 时间:2023年6月6日09:38:48
    # src_dir:机台bmp数据根目录
    # dst_dst: G:\DefectDataCenter\ParseData ,解析后的数据位置
    # 功能：提取汇总src_dir的数据

    #case 1
    # src_dir =r'G:\DefectDataCenter\原始_现场分类数据\MXB\AI分类有异常的数据\Lot\sct_asi_0401'
    # dst_dir = r'G:\DefectDataCenter\ParseData\Classifier\Color_Difference_issue\sct_asi_0401'
    #case2
    # src_dir =r'G:\DefectDataCenter\原始_现场分类数据\MXB\新增数据训练后分类正常数据\20230525193915'
    # dst_dir = r'G:\DefectDataCenter\ParseData\Classifier\Color_Difference_issue\20230525193915'

    folders = os.listdir(src_dir)
    for folder in folders:
        sub_dir = os.path.join(src_dir,folder,"1")
        dst_folder = os.path.join(dst_dir,folder)
        os.makedirs(dst_folder,exist_ok=True)

        bmp_folders = os.listdir(sub_dir)
        for bmp_folder in bmp_folders:
            bmps_dir = os.path.join(sub_dir,bmp_folder)
            bmps = os.listdir(bmps_dir)
            for bmp in bmps:
                if bmp.endswith("bmp"):
                    src_path = os.path.join(bmps_dir,bmp)
                    dst_path =os.path.join(dst_folder,bmp)
                    # shutil.copy(src_path, dst_path)
                    shutil.move(src_path,dst_path)





if __name__ == '__main__':
#sct_asi_0401 :AI分类有异常的数据

    src_dir =r'G:\DefectDataCenter\原始_现场分类数据\MXB\新增数据训练后分类正常数据\20230528203716'
    dst_dir = r'G:\DefectDataCenter\ParseData\Classifier\Color_Difference_issue\20230528203716'

    extract_bmp4_WSD(src_dir,dst_dir)
