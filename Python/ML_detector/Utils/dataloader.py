import cv2, os
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from Utils.config import Config

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        # self.num_classes = num_classes
        self.length = len(self.annotation_lines)
        # self.train = train

        # self.vis_list = list(range(self.vis_num))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, box, img_path = self.get_random_data(index)
        image = np.transpose(image / 255.0, (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            ##将box[x1,y1,x2,y2]-->box[x_center,y_center,w,h]
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box, img_path
    def get_random_data(self, index):
        line = self.annotation_lines[index].split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = cv2.imdecode(np.fromfile(line[0], dtype=np.uint8), 1)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        ih, iw, _ = image.shape  # 图片原始尺寸
        h, w = self.input_shape  # 需要resize的尺寸
        # ------------------------------------------#
        #   对图像进行缩放
        # ------------------------------------------#
        image = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        box[:, [0, 2]] = box[:, [0, 2]] * w / iw
        box[:, [1, 3]] = box[:, [1, 3]] * h / ih
        return image, box, line[0]

    def visualize_data(self,vis_num):
        vis_num = min(vis_num,self.length) #True
        dst_dir = r'data/check_trian_img'
        os.makedirs(dst_dir, exist_ok=True)
        for vis_i in range(vis_num):
            image, box, img_p = self.get_random_data(vis_i)
            img_n = os.path.basename(img_p)
            for bx_f in box:
                # #使img内部内存连续
                bx = [int(x) for x in bx_f]
                image = np.ascontiguousarray(image)
                cv2.rectangle(image, (bx[0], bx[1]), (bx[2], bx[3]), (100, 210, 20), 2)
            dst_img_p = os.path.join(dst_dir, img_n)
            cv2.imwrite(dst_img_p, image)

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    img_paths = []
    for img, box, img_path in batch:
        images.append(img)
        bboxes.append(box)
        img_paths.append(img_path)
    images = np.array(images)
    return images, bboxes, img_paths


if __name__ == '__main__':
    project = "COT_Raw"  ## LMK, HW,VOC,DSW_random ,COT_Raw ;COT_Raw ; DSW
    cfg = Config(project)
    size_input = [cfg.ImageSizeY, cfg.ImageSizeX]
    train_ls = cfg.train_lines
    vis_num = 10
    ##创建数据集，General ,查看训练标签（缩放后的）是否正确
    train_set = YoloDataset(train_ls,size_input)
    train_set.visualize_data(vis_num)
    gen = DataLoader(train_set, shuffle=True, batch_size=8, num_workers=0,
                     pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
    for iteration, batch in enumerate(gen):
        print("iteration：{}".format(iteration))