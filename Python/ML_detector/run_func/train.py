#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import torch.optim as optim
from torch.utils.data import DataLoader
from Utils.callbacks import LossHistory
from Utils.dataloader import YoloDataset ,yolo_dataset_collate
from Utils.utils_fit import fit_one_epoch
from Utils.config import Config
# # 避免 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def yolo4_train():
    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    project = "COT_Raw"  ## LMK, HW,VOC,DSW_random ,COT_Raw ;COT_Raw ; DSW
    cfg = Config(project)
    cls_n = len(cfg.class_names)
    size_input = [cfg.ImageSizeY, cfg.ImageSizeX]

    train_ls = cfg.train_lines
    val_ls = cfg.val_lines
    batch_sz = cfg.batch_size
    worker_n = cfg.num_workers
    lr1 = cfg.lr[0]
    lr2 = cfg.lr[1]
    yolo_loss = cfg.yolo_loss
    loss_history = LossHistory("../logs/")
    ##创建数据集，General
    train_set = YoloDataset(train_ls,size_input,cls_n, train=False)
    val_set = YoloDataset(val_ls, size_input, cls_n, train=False)
    gen = DataLoader(train_set, shuffle=True, batch_size=batch_sz, num_workers=worker_n,
                     pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_set, shuffle=True, batch_size=batch_sz, num_workers=worker_n,
                         pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    # 创建优化器
    optimizer = optim.Adam(cfg.model_train.parameters(), lr1, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    for param in cfg.model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(0, cfg.fepoch):
        fit_one_epoch(cfg, yolo_loss, loss_history, optimizer, epoch, cfg.fepoch, gen, gen_val)
        lr_scheduler.step()
    # ------------------------------------#
    #   解冻训练
    # ------------------------------------#
    optimizer = optim.Adam(cfg.model_train.parameters(), lr2, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    for param in cfg.model.backbone.parameters():
        param.requires_grad = True
    for epoch in range(cfg.fepoch, cfg.all_epoch):
        fit_one_epoch(cfg, yolo_loss, loss_history, optimizer, epoch, cfg.all_epoch, gen, gen_val)
        lr_scheduler.step()

if __name__ == "__main__":
    yolo4_train()
