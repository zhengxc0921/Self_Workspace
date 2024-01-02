import os
import pandas as pd

import openpyxl
class Calc_Bed():
    ##爱心送货表结果计算
    def __init__(self):
        self.src_dir = r'D:\bed_color\src_d\2024_1_2'
        self.dst = r'D:\bed_color\rst\2024_1_2'
        os.makedirs(self.dst,exist_ok=True)
        self.size_info = ['90', '120', '150', '180']
        self.color_info = ['蓝灰', '豆绿', '卡其', '浅灰', '雪白', '玉色']

        self.pillow_price = 13
        self.bed_price = {'90*200':32,'120*200':36,'150*200':40,'180*200':44}

        self.bed_rst_map = {}
        self.pillow_rst_map = {}
        for c in self.color_info:
            for s in self.size_info:
                k = c + '_' + s + "*200"
                self.bed_rst_map[k] = 0
                self.pillow_rst_map[k] = 0

    def extract_txts(self,src_txt):

        data = pd.read_excel(io=src_txt, header=None)
        for index, row in data.iterrows():
            # 规格
            s, c = '', ''
            for tmp_s in self.size_info:
                if tmp_s in row[1]:
                    s = tmp_s
            # 颜色
            for tmp_c in self.color_info:
                if tmp_c in row[2]:
                    c = tmp_c
            tmp_k = c + '_' + s + "*200"
            # 数量
            tmp_n = int(row[3])
            self.bed_rst_map[tmp_k] += tmp_n
            # 是否有枕套
            if '三件套' in row[2]:
                self.pillow_rst_map[tmp_k] += tmp_n

        return

    def save_rst2excel(self,excel_name):
        # 创建一个新的Excel工作簿
        workbook = openpyxl.Workbook()
        # 获取默认的工作表
        sheet = workbook.active
        # 写入颜色、规格数据
        colors_rst=[]
        sheet['A1'] = '颜色'
        sheet['B1'] = '规格'
        sheet['C1'] = '床笠数量'
        sheet['D1'] = '枕套数量'
        for k,v in self.bed_rst_map.items():
            if int(v) !=0:
                c = k.split('_')[0]
                s = k.split('_')[1]
                pillow_n = self.pillow_rst_map[k]
                if pillow_n==0:
                    pillow_n=''
                d1 = [c,s,v,pillow_n]
                sheet.append(d1)
                colors_rst.append(d1)
        # 从color_rst中提取出 尺寸、数量，计算金额,枕头套数量  --》size_rst_map，pillow_num
        size_rst_map = {}
        pillow_num = 0
        bed_num = 0
        for c_rst in colors_rst:
            size = c_rst[1]
            tmp_bed_num = c_rst[2]
            bed_num +=tmp_bed_num
            tmp_pillow = c_rst[3]
            if tmp_pillow!='':
                pillow_num +=  int(c_rst[3])
            if size not in size_rst_map:
                size_rst_map[size]=tmp_bed_num
            else:
                size_rst_map[size] += tmp_bed_num
        #写入表单，小计结果
        sheet.append(['小计数量',' ',bed_num,pillow_num])
        #写入表单, 将size_rst_map，pillow_num 计算金额
        sheet.append([' '])
        total_value = 0
        size_rst_title = ['尺寸', '数量', '金额']
        sheet.append(size_rst_title)
        for sk,sv in size_rst_map.items():
            single_value = sv*self.bed_price[sk]
            total_value +=single_value
            tmp_size_rst = [sk,sv,single_value]
            sheet.append(tmp_size_rst)

        #枕头套数量
        pillow_rst_title=['枕头套数量','金额(元)']
        sheet.append(pillow_rst_title)
        pillow_value = pillow_num*self.pillow_price
        total_value += pillow_value
        tmp_pillow_rst = [pillow_num,pillow_value]
        sheet.append(tmp_pillow_rst)
        #写入总金额
        sheet.append([' '])
        sheet.append(['床笠与枕套总金额(元)',total_value])
        # 保存工作簿
        dst_p =os.path.join(self.dst,excel_name)  #excel_name  'example.xlsx'
        workbook.save(dst_p)

    def extract_dir(self):

        files = os.listdir(self.src_dir)
        for file in files:
            file_p =os.path.join(self.src_dir,file)
            self.extract_txts(file_p)
            rst_file_n = "送货表结果_"+file
            self.save_rst2excel(rst_file_n)
        # self.save_rst2excel('example.xlsx')
        return
        #将数据保存到excel中


if __name__ == '__main__':
    A = Calc_Bed()
    A.extract_dir()


# def extract_txts():
#     src_txt = r'D:\word_extract.xlsx'
#     data = pd.read_excel(io=src_txt,header=None)
#     size_info = ['90','120','150','180']
#     color_info = ['蓝灰','豆绿','卡其','浅灰','雪白','玉色']
#
#     bed_rst_map = {}
#     pillow_rst_map = {}
#     for c in color_info:
#         for s in size_info:
#             k = c+'_'+s+"*200"
#             bed_rst_map[k]=0
#             pillow_rst_map[k] = 0
#
#
#     for index,row in data.iterrows():
#         #规格
#         s,c = '',''
#         for tmp_s in  size_info:
#             if tmp_s in row[1]:
#                 s = tmp_s
#         #颜色
#         for tmp_c in  color_info:
#             if tmp_c in row[2]:
#                 c = tmp_c
#         tmp_k = c+'_'+s+"*200"
#         #数量
#         tmp_n = int(row[3])
#         bed_rst_map[tmp_k] +=tmp_n
#         #是否有枕套
#         if '三件套' in row[2]:
#             pillow_rst_map[tmp_k] += tmp_n
#
#     return



