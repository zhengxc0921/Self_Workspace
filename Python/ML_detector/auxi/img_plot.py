"""
File_name: img_plot
Author: frank_zhengxc
Description: Draw img quickly from numpy data
Create time: 2021-05-06
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt

def numpy_plot(numpy_array,img_id,boxes=None):
    """
    Plot img from numpy_array while the code is debugging
      - squeeze the dimension and select the "img_id" channel to show
      - if boxes is not none ,add the boxs to the img

    Parameters
    ----------
    numpy_array : (...,img_id,H,W)
    boxes:(1,boxs_num,[left,top,right,down])
    """
    # imgs = numpy_array.squeeze()
    imgs = numpy_array
    img = imgs[img_id]
    plt.figure()
    plt.imshow(np.transpose(img,(1,2,0)))
    if boxes is not None:
        for j in range(boxes[0].shape[0]):
            i = 0
            plt.gca().add_patch(plt.Rectangle(xy=(boxes[i][j][0], boxes[i][j][1]),
                                              height=boxes[i][j][3] - boxes[i][j][1],
                                              width=boxes[i][j][2] - boxes[i][j][0],
                                              fill=False, linewidth=2, edgecolor="red"))


    filePath = './img/test_'+str(img_id)+'.png'
    plt.savefig(filePath)
    plt.show()

def yolo_numpy_plot(numpy_array,img_id,boxes=None):
    """
    Plot img from numpy_array while the code is debugging
      - squeeze the dimension and select the "img_id" channel to show
      - if boxes is not none ,add the boxs to the img

    Parameters
    ----------
    numpy_array : (...,img_id,H,W)
    boxes:(1,boxs_num,[left,top,right,down])
    """
    # imgs = numpy_array.squeeze()
    imgs = numpy_array
    img = imgs[img_id]
    c, h,w = img.shape

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('TkAgg')
    plt.figure()
    plt.imshow(np.transpose(img,(1,2,0)))
    if boxes is not None:
        for j in range(boxes[0].shape[0]):
            i = 0
            box_x = (boxes[i][j][0]-boxes[i][j][2]/2)*w
            box_y = (boxes[i][j][1] - boxes[i][j][3] / 2)*h
            box_h =  boxes[i][j][3]*h
            box_w = boxes[i][j][2]*w


            plt.gca().add_patch(plt.Rectangle(xy=(box_x, box_y),
                                              height=box_h,
                                              width=box_w,
                                              fill=False, linewidth=2, edgecolor="red"))
    os.makedirs('./img',exist_ok=True)
    filePath = './img/test_'+str(img_id)+'.png'
    plt.savefig(filePath)
    plt.close()
    # plt.show()

def tensor_plot(tensor_array,img_id=0,img_title='img',boxes=None,savepath=None):
    """
    Plot img from numpy_array while the code is debugging
      - squeeze the dimension and select the "img_id" channel to show
      - if boxes is not none ,add the boxs to the img

    Parameters
    ----------
    numpy_array : (...,img_id,H,W)
    boxes:(1,boxs_num,[left,top,right,down])
    """
    # img_title = 'input_img-template'
    numpy_array = np.array(tensor_array)
    imgs = numpy_array.squeeze()
    img = imgs[img_id]
    plt.figure()
    plt.imshow(img)
    plt.title(img_title)
    plt.colorbar()
    if savepath is not None:
        savepath = os.path.join(savepath,img_title)
        plt.savefig(savepath+'.png')
    if boxes is not None:
        for j in range(boxes.shape[1]):
            i = 0
            plt.gca().add_patch(plt.Rectangle(xy=(boxes[i][j][0], boxes[i][j][1]),
                                              height=boxes[i][j][3] - boxes[i][j][1],
                                              width=boxes[i][j][2] - boxes[i][j][0],
                                              fill=False, linewidth=2, edgecolor="red"))
    plt.show()

def batch_plot(bat_imgs,bat_boxes,baox_i=0,savepath=None):
    #case1:
    # from auxilliary.img_plot import numpy_plot,batch_plot
    # savepath=r'E:\项目代码\缺陷检测\fpn_faster-rcnn-pytorch-master_DSW\auxilliary\debug'
    # batch_plot(x.cpu(),rois[np.newaxis,:,:],savepath=savepath)
    img_tf = np.transpose(bat_imgs[0], (1, 2, 0))
    img = np.array(img_tf)
    boxes = np.array(bat_boxes[baox_i]).reshape(1, -1, 4)
    plt.figure()
    plt.imshow(img)
    plt.title('img_1')

    if boxes is not None:
        for j in range(boxes.shape[1]):
            i = 0
            plt.gca().add_patch(plt.Rectangle(xy=(boxes[i][j][0], boxes[i][j][1]),
                                              height=boxes[i][j][3] - boxes[i][j][1],
                                              width=boxes[i][j][2] - boxes[i][j][0],
                                              fill=False, linewidth=2, edgecolor="red"))
    if savepath is not None:
        savepath = os.path.join(savepath,'img_1')
        plt.savefig(savepath+'.png')
    plt.show()

def muti_batch_plot(muti_batch_imgs, boxes,batch_id=0):
    img_tf = np.transpose(muti_batch_imgs[batch_id], (1, 2, 0))
    img = np.array(img_tf)
    plt.figure()
    plt.imshow(img)
    if boxes is not None:
        for j in range(boxes[batch_id].shape[0]):
            i = batch_id
            plt.gca().add_patch(plt.Rectangle(xy=(boxes[i][j][0], boxes[i][j][1]),
                                              height=boxes[i][j][3] - boxes[i][j][1],
                                              width=boxes[i][j][2] - boxes[i][j][0],
                                              fill=False, linewidth=2, edgecolor="red"))
    plt.show()

##check_train_box
def check_train_box( targets,img_paths,class_names):
    import cv2 , colorsys,os
    dst_dir = r'./img'
    os.makedirs(dst_dir, exist_ok=True)

    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    def visualize_bbox(img, bbox, class_name, thickness=2):
        """Visualizes a single bounding box on the image"""
        # x_min, y_min, w, h = bbox
        y_min, x_min, y_max, x_max = bbox
        img = np.ascontiguousarray(img)
        cls_index = class_names.index(class_name)

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=colors[cls_index],
                      thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        y_min_n = y_min - int(1.3 * text_height)
        x_max_n = x_max + text_width
        cv2.rectangle(img, (int(x_min), int(y_min_n)), (int(x_max), int(y_min)), color=(255, 255, 255), thickness=-1)

        cv2.putText(
            img,
            text='{}'.format(class_name),
            org=(int(x_min), int(y_min) - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=colors[cls_index],
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize(img, top_label , top_boxes):
        for i, c in list(enumerate(top_label)):
            class_name = class_names[int(c)]
            bbox = top_boxes[i]
            img = visualize_bbox(img, bbox, class_name)

        return img
    for i,img_p in enumerate(img_paths):
        image = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), 1)
        img_n = os.path.basename(img_p)
        top_boxes = targets[i][:, :4]
        top_label =  targets[i][:,4]
        img = visualize(image, top_label, top_boxes)
        dst_img_p = os.path.join(dst_dir,img_n)
        cv2.imwrite(dst_img_p,img)

    return

