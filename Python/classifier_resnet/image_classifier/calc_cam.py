# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

img_size = (224,224)



def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,img_size)

    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir,n):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir,str(n) +"_cam.jpg")
    path_raw_img = os.path.join(out_dir, str(n)+"_raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #
    # print("weight: ",weights[:10])
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, img_size)
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
    from net_model.resnet import ResNet18
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_img = r'G:\DefectDataCenter\DeepLearningDataSet\Output\DXC_delta_c_r_128\defect'
    model_name = r'cpu_DXC_delta_c_r_128.pkl'
    # model_name = r'cpu_GAT.pkl'
##
    # path_net = os.path.join(BASE_DIR, "..", "..", "Data", "net_params_72p.pkl")
    output_dir = r'E:\SoftWareInstaller\python_demo_20201228\CAM_d'
    classes =  ('defect','normal')

    net =ResNet18(num_classes=2)
    path_net = os.path.join(r"..\result", model_name)
    net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.blk4.register_forward_hook(farward_hook)
    net.blk4.register_backward_hook(backward_hook)

    for n, img_n in enumerate(os.listdir(folder_img)):
        fmap_block = list()
        grad_block = list()
        path_img = os.path.join(folder_img, img_n)
        # 图片读取；网络加载
        # img = cv2.imread(path_img, 1)  # H*W*C

        # img = cv2.imdecode(np.fromfile(path_img, dtype=np.uint8),cv2.IMREAD_UNCHANGED)  # H*W*C
        img = cv2.imdecode(np.fromfile(path_img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # 灰度图转换为彩图
        if len(img.shape) == 3:
            img = img
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_input = img_preprocess(img)
        # forward
        output = net(img_input)
        idx = np.argmax(output.cpu().data.numpy())
        print("{} predict: {}".format(img_n,classes[idx]))
        print("{} output: {}".format(img_n,output))

        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        # print("class_loss:",class_loss)
        class_loss.backward()

        # 生成cam
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        # print("grads_val: ",grads_val[0])
        # print("sum(sum(grads_val))： ",sum(sum(grads_val)))
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        cam = gen_cam(fmap, grads_val)

        # 保存cam图片
        img_show = np.float32(cv2.resize(img, img_size)) / 255
        show_cam_on_image(img_show, cam, output_dir,img_n)
