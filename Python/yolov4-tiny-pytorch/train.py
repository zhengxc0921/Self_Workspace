#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader


from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils_fit import fit_one_epoch


from utils.config import Config
# # 避免 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def yolo4_train():
    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    project = "COT_Raw"  ## LMK, HW,VOC,DSW_random
    cfg = Config(project)
    yolo_loss = YOLOLoss(cfg.anchors, cfg.num_classes, cfg.input_shape, cfg.calc_device, cfg.anchors_mask).to(
        cfg.calc_device)
    loss_history = LossHistory("logs/")

    ##创建数据集，General
    train_dataset = YoloDataset(cfg.train_lines, cfg.input_shape, cfg.num_classes, train=False)
    val_dataset = YoloDataset(cfg.val_lines, cfg.input_shape, cfg.num_classes, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                     pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    # 创建优化器
    optimizer = optim.Adam(cfg.model_train.parameters(), cfg.lr[0], weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    if cfg.epoch_step == 0 or cfg.epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
    for param in cfg.model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(0, cfg.fepoch):
        fit_one_epoch(cfg, yolo_loss, loss_history, optimizer, epoch, cfg.fepoch, gen, gen_val)
        lr_scheduler.step()
    # ------------------------------------#
    #   解冻训练
    # ------------------------------------#
    optimizer = optim.Adam(cfg.model_train.parameters(), cfg.lr[1], weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    for param in cfg.model.backbone.parameters():
        param.requires_grad = True
    for epoch in range(cfg.fepoch, cfg.all_epoch):
        fit_one_epoch(cfg, yolo_loss, loss_history, optimizer, epoch, cfg.all_epoch, gen, gen_val)
        lr_scheduler.step()

def yolo4_traint():
    print("find train file")
if __name__ == "__main__":
    # pass
    yolo4_train()


