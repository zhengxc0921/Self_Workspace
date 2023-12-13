import os

import torch
from tqdm import tqdm
from Utils.utils import get_lr


def fit_one_epoch(cfg, yolo_loss, loss_history, optimizer, epoch, Epoch, gen, gen_val):
    model_train = cfg.model_train
    model = cfg.model
    epoch_step = cfg.epoch_step
    epoch_step_val = cfg.epoch_step_val
    calc_device = cfg.calc_device
    print('epoch_step_val：{}'.format(epoch_step_val))
    loss = 0
    val_loss = 0
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            images, targets, img_paths = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).to(calc_device, dtype=torch.float32)
            optimizer.zero_grad()
            ##float32
            outputs = model_train(images)
            loss_value_all = 0
            num_pos_all = 0
            targets = [torch.from_numpy(ann).to(calc_device, dtype=torch.float32) for ann in targets]
            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos
            loss_value = loss_value_all / num_pos_all
            # print("loss_value：{}".format(loss_value))
            loss_value.backward()
            optimizer.step()
            loss += loss_value.item()
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets,_ = batch[0], batch[1],batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).to(calc_device, dtype=torch.float32)
                targets = [torch.from_numpy(ann).to(calc_device, dtype=torch.float32) for ann in targets]
                # ----------------------#
                #   清零梯度
                # ----------------------#
                optimizer.zero_grad()
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(images)
                loss_value_all = 0
                num_pos_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                    num_pos_all += num_pos
                loss_value = loss_value_all / num_pos_all
            val_loss += loss_value.item()
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)

    print('Finish Validation')
    # loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    #每8个epoch更新一个epoch
    if Epoch == epoch + 1 or epoch % 8 == 0:
        torch.save(model.state_dict(), cfg.pth_dst)
