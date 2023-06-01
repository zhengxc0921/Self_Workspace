

from image_classifier.argu import *

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
    class_vec = torch.sum(one_hot * ouput_vec)  # one_hot = 11.8605

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

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam

def show_cam_on_image(img, mask, out_dir,i,img_fix):
    k = 3
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)*k
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, img_fix+"_cam.jpg")
    # path_raw_img = os.path.join(out_dir, img_fix+"_raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    # cv2.imwrite(path_raw_img, np.uint8(255 * img))


class NetOperate:
    def __init__(self,img_argu):
        self.img_argu = img_argu
        #迭代参数
        self.epochs = 20
        self.best_epoch = 0
        self.val_acc = 0.0
        self.best_acc = 0.0

    def calc_loss(self, logits, y, class_ratio):
        if type(logits) == tuple:
            # 计算googlenet的损失函数
            loss0 = self.img_argu.criteon(logits[0], y, class_ratio)
            loss1 = self.img_argu.criteon(logits[1], y, class_ratio)
            loss2 = self.img_argu.criteon(logits[2], y, class_ratio)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        else:
            # 计算其他net的损失函数
            loss = self.img_argu.criteon(logits, y, class_ratio)
        return loss

    def model_train(self):
        # 模型训练主程序
        tt1 = time.time()
        for epoch in range(self.epochs):
            for step, (x, y, image_ID, image_path,normal_blob) in enumerate(self.img_argu.train_loader):
                x, y = x.to(self.img_argu.calc_device), y.to(self.img_argu.calc_device)
                self.img_argu.model.train()  # 指定当前模型是在训练（使用BatchNormalizetion()和Dropout()）

                logits = self.img_argu.model(x)
                loss = self.calc_loss(logits, y, self.img_argu.class_num_ratio)  # 计算损失函数
                self.img_argu.optimizer.zero_grad()  # 如不清零，那么grad就同上一个mini-batch有关
                loss.backward()  # 计算反向梯度
                self.img_argu.optimizer.step()  # optimizer更新参数空间需要基于反向梯度
            if epoch % 2 == 0 or epoch==self.epochs:
                self.model_evaluate(self.img_argu.val_loader)
                if self.val_acc > self.best_acc:
                    self.best_epoch = epoch
                    self.best_acc = self.val_acc
                    torch.save(self.img_argu.model.state_dict(), self.img_argu.best_model_path)
            # 每训练10次保存模型
            if epoch % 5 == 0:
                tmp_model_name =  self.img_argu.model_name  + '_epoch_' + str(epoch) + '.pkl'
                tmp_model_path = self.img_argu.model_dst_dir + tmp_model_name
                torch.save(self.img_argu.model.state_dict(), tmp_model_path)
        tt2 = time.time()
        # 保存最终训练模型
        print('best acc:', self.best_acc, 'best epoch:',   self.best_epoch)
        return

    def calc_recall(self,pre,target):
        pre_right = [x for (x,y) in zip(pre,target) if x==y]
        target_types = set(target)
        pre_dict = {}
        recall_dict = {}
        error_detection = {}
        for t_type in target_types:
            #calc precision:正确判A_num/所有判类A_num
            if pre.count(t_type)==0:
                pre_dict[t_type]=0
            elif target.count(t_type)==0:
                recall_dict[t_type]=0
            else:
                pre_dict[t_type] = pre_right.count(t_type)/pre.count(t_type)
                #calc recall:正确判A_num/所有真类A_num
                recall_dict[t_type] = pre_right.count(t_type)/target.count(t_type)
            #calc_error_detection :误判类A_num/所有真非类A_num
            if len(target)-target.count(t_type)> 0:
                error_detection[t_type] = (pre.count(t_type)- pre_right.count(t_type)) /(len(target)-target.count(t_type))
            else:
                error_detection[t_type] = 0
        return pre_dict,recall_dict,error_detection

    def img_classifier_save(self,logits, image_ID):
        # 获取标签
        label2name = {v:k.split("\\")[-1] for (k,v) in self.img_argu.data_classifer.name2label.items()}
        for step, image_raw_dir in enumerate(image_ID):
            img_dect_type = label2name[logits[step]]
            dist_path = os.path.join(self.img_argu.pre_dst_dir,img_dect_type)
            # if logits[step] == 0:
            shutil.copy(image_ID[step], dist_path)
        return



    def cv_load_image(self, img_read):
            adaptive_size = [224, 224]
            # 灰度图转换为彩图
            if len(img_read.shape) == 3:
                img_cv_gray = img_read / 255.0
            else:
                img_cv_gray = cv2.cvtColor(img_read, cv2.COLOR_GRAY2BGR) / 255.0
            img_cv_resize = cv2.resize(img_cv_gray, tuple(adaptive_size), interpolation=cv2.INTER_LINEAR)
            # # 图像裁剪之后的结果
            img_cv_resize_np_f64 = np.transpose(img_cv_resize, [2, 0, 1])
            img_cv_resize_np = np.array(img_cv_resize_np_f64, dtype='float32')
            img_tensor = torch.tensor(img_cv_resize_np).unsqueeze(0)
            return img_tensor

    def model_predict(self, dst_dir,src_dir):
        img_list= os.listdir(src_dir)
        for img_n in img_list:
            img_path = os.path.join(src_dir,img_n)
            img_read = cv2.imread(img_path)
            x = self.cv_load_image(img_read)
            with torch.no_grad():  # 设置梯度不发生变化
                logits = self.img_argu.model(x)
                pred = int(np.array(logits.argmax(dim=1)) ) # 获取预测值
                src_img = os.path.join(src_dir, img_n)
                dst_path = os.path.join(dst_dir, str(self.img_argu.label2class[pred]), img_n)
                shutil.copy(src_img, dst_path)

    def model_cam(self,dst_dir,loader):
        # 模型评估
        self.img_argu.iter_p_dict['correct'] = 0
        types =  ['defect','normal']
        def backward_hook(module, grad_in, grad_out):
            grad_block.append(grad_out[0].detach())
        def farward_hook(module, input, output):
            fmap_block.append(output)
        # 注册hook
        self.img_argu.model.blk3.register_forward_hook(farward_hook)
        self.img_argu.model.blk3.register_backward_hook(backward_hook)
        i = 0
        for x, y, image_ID, images_path, normal_blob in loader:
            fmap_block = list()
            grad_block = list()
            img = images_path[0].split("\\")[-1]
            x, y = x.to(self.img_argu.calc_device), y.to(self.img_argu.calc_device)
            self.img_argu.iter_p_dict['loader_num'] += 1
            logits = self.img_argu.model(x)
            pred = logits.argmax(dim=1)  # 获取预测值
            # print("predict: {}".format(types[pred]))
            # backward
            self.img_argu.model.zero_grad()
            class_loss = comp_class_vec(logits)
            class_loss.backward()
            # 生成cam
            grads_val = grad_block[0].cpu().data.numpy().squeeze()
            fmap = fmap_block[0].cpu().data.numpy().squeeze()
            cam = gen_cam(fmap, grads_val)
            # 保存cam图片
            src_img = images_path[0]
            if pred == 0:
                dst_dir_c = os.path.join(dst_dir, types[0])
                dst_path = os.path.join(dst_dir_c, img)
                shutil.copy(src_img, dst_path)
            else:
                dst_dir_c = os.path.join(dst_dir, types[1])
                dst_path = os.path.join(dst_dir_c, img)
                shutil.copy(src_img, dst_path)
            img_show = np.float32( x.squeeze().permute(1,2,0))
            img_fix = img.split(".")[0]
            show_cam_on_image(img_show, cam, dst_dir_c,i,img_fix)

    def model_evaluate(self, loader):
        # 模型评估
        self.img_argu.model.eval()  # 指定当前模型是在训练（#不使用BatchNormalization()和Dropout()）
        correct_num = 0
        predict_results = []
        target_results = []
        for x, y, image_ID, images_path,normal_blob in loader:
            # images_path_list.extend(images_path)
            x, y = x.to(self.img_argu.calc_device), y.to(self.img_argu.calc_device)
            with torch.no_grad():  # 设置梯度不发生变化
                logits = self.img_argu.model(x)
                pred = logits.argmax(dim=1)  # 获取预测值
                predict_results.extend(np.array(pred))
                target_results.extend(np.array(y))
                correct_num+= torch.eq(pred,y).sum().float().item()  # 累计loader1次的正确预测数量bool(self.train_loader)
        pre_dict, recall_dict, error_detection= self.calc_recall(predict_results,target_results)
        self.val_acc = correct_num/ len(loader.dataset)
        print('val_acc:', self.val_acc )
        print("pre_dict:",pre_dict,"recall_dict:",recall_dict,"error_detection:",error_detection)

    def model_load(self,model_path=None):
        if os.path.exists(model_path):
            self.img_argu.model.load_state_dict(torch.load(model_path))
            print('模型加载成功！')
        else:
            print('无保存模型，将从头开始训练！')
            self.model_train()
        return


if __name__ == '__main__':
    pass
