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


    filePath = './img/test_'+str(img_id)+'.png'
    plt.savefig(filePath)
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

# from auxilliary.img_plot import tensor_plot
# for i in range(20):
#     tensor_plot(h.cpu(),i,img_title='img_'+str(i),
#     savepath=r'E:\项目代码\缺陷检测\fpn_faster-rcnn-pytorch-master_DSW\DSW_data\DSW_analytical\Out_Feature_减模板')

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