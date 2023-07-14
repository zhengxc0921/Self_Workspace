import os ,cv2,shutil,time
import numpy as np
from sklearn.cluster import DBSCAN
class ImgCluster:
    def __init__(self,src_dir,dst_dir,Eeps,pn):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.Eeps =Eeps
        self.pn = pn
        self.img_size = (16, 16)
        self.delet_ratio = 0.5
        self.size_thres = 3

        self.all_images = []
        self.all_mean_images = []
        self.all_paths = []
        self.get_imglists()

    def get_imglists(self):
        invalid_folder = os.path.join(self.dst_dir,self.pn,'-2')
        os.makedirs(invalid_folder,exist_ok=True)

        imgn_list = os.listdir(self.src_dir)
        for img_n in imgn_list:
            path = os.path.join(self.src_dir,img_n)
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            h,w = img.shape
            SMALL_IMG = h/w >self.size_thres or (w/h>self.size_thres)
            if SMALL_IMG :
                src_img_path = os.path.join(self.src_dir,img_n)
                invalid_img_path = os.path.join(invalid_folder,img_n)
                shutil.copy(src_img_path,invalid_img_path)
            else:
                img_r = cv2.resize(img,self.img_size)
                img_arr = img_r.reshape(-1,)
                img_m = img_arr / 255  ##此种处理方式无法处理色差
                self.all_mean_images.append(img_m)
                self.all_paths.append(img_n)

    def DB_cluster(self):
        clt = DBSCAN(eps=self.Eeps , min_samples=60).fit(self.all_mean_images)
        labelIDs = np.unique(clt.labels_)
        for labelID in labelIDs[1:]:
            dst_dir = os.path.join(self.dst_dir,self.pn,str(labelID))
            os.makedirs(dst_dir,exist_ok=True)
            idxs = np.where(clt.labels_ == labelID)[0]
            delet_n = int(self.delet_ratio*len(idxs))
            delet_idxs = np.random.choice(idxs,delet_n,replace=False)

            for idx in delet_idxs:
                img_n = self.all_paths[idx]
                src_img_path = os.path.join(self.src_dir,img_n)
                dst_img_path = os.path.join(dst_dir,img_n)
                shutil.copy(src_img_path,dst_img_path)
        return

def pt1(src_dir,dst_dir,Eeps,pn):
    t1 = time.time()
    A = ImgCluster(src_dir,dst_dir,Eeps,pn)
    A.DB_cluster()
    t2 = time.time()
    print("t1-t2:", t2 - t1)

def pt3(src_dir):
    src_dir =src_dir
    dst_dir = r'G:\DefectDataCenter\Test\ImgCluster'
    Eeps = 1.2
    pn = 'SPA90'
    t1 = time.time()
    A = ImgCluster(src_dir,dst_dir,Eeps,pn)
    A.DB_cluster()
    t2 = time.time()
    print("t1-t2:", t2 - t1)
    

def pt4(Eeps):
    src_dir =r'G:\DefectDataCenter\Test\Src\90'
    dst_dir = r'G:\DefectDataCenter\Test\ImgCluster'
    Eeps = Eeps
    pn = 'SPA90'
    t1 = time.time()
    A = ImgCluster(src_dir,dst_dir,Eeps,pn)
    A.DB_cluster()
    t2 = time.time()
    print("t1-t2:", t2 - t1)


def pt2():
    src_dir =r'G:\DefectDataCenter\Test\Src\90'
    dst_dir = r'G:\DefectDataCenter\Test\ImgCluster'
    Eeps = 1.2
    pn = 'SPA90'
    t1 = time.time()
    A = ImgCluster(src_dir,dst_dir,Eeps,pn)
    A.DB_cluster()
    t2 = time.time()
    print("t1-t2:", t2 - t1)



if __name__ == '__main__':
    src_dir =r'G:\DefectDataCenter\Test\Src\90'
    dst_dir = r'G:\DefectDataCenter\Test\ImgCluster'
    Eeps = 1.2
    pn = 'SPA90'
    pt1(src_dir,dst_dir,Eeps,pn)
