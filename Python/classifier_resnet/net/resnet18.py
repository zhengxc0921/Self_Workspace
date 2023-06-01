from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # 下面是瓶颈层
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)  # padding=1保证输出大小一致
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:  # 保存通道一致，才可以相加
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out))

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        # self.extra(x)
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):  # 构建ResNet18模型
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=0),  # raw
        #     # nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=0),
        #     # 3：表示的是输入的通道数，由于是RGB型的，所以通道数是3
        #     # 16：表示的是输出的通道数，设定输出通道数的16
        #     # 输出大小N公式：（W-F+2*P）/S+1
        #     nn.BatchNorm2d(32)  # 特征缩放，更利于数据搜索最优解。把数据按通道缩放到N~(0,1)
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,bias=False),  # raw
            # nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=0),
            # 3：表示的是输入的通道数，由于是RGB型的，所以通道数是3
            # 16：表示的是输出的通道数，设定输出通道数的16
            # 输出大小N公式：（W-F+2*P）/S+1
            nn.BatchNorm2d(32)  # 特征缩放，更利于数据搜索最优解。把数据按通道缩放到N~(0,1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(32, 64, stride=3)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(64, 128, stride=3)
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(128, 256, stride=2)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(256, 512, stride=2)
        # [b, 256, 7, 7]
        self.outlayer = nn.Linear(512 * 2 * 2, num_classes)  # 256*3*3:由输入张量self.blk4的形状决定
        # 3*3必须经过图片->conv1->blk1->...blkn(具体步骤参考函数forward),一步步计算

    def forward(self, x):
        """
        :完成正向网络的串联
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)  # 作用类似numpy中的reshape（x.size(0), -1）

        x = self.outlayer(x)
        x = F.softmax(x,dim=1)
        return x

def resnet18(num_classes):
    model = ResNet18(num_classes=num_classes)
    # 获取特征提取部分
    features = list([model.conv1, model.maxpool, model.blk1, model.blk2, model.blk3, model.blk4])
    # 获取分类部分
    classifier = list([model.outlayer])
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier

if __name__ == '__main__':
    import torch
    import inspect
    from net_model.gpu_mem_track import MemTracker  # 引用显存跟踪代码
    device = torch.device('cuda:0')
    frame = inspect.currentframe()
    gpu_tracker = MemTracker(frame)  # 创建显存检测对象
    gpu_tracker.track()
    # x = torch.randn([1, 3, 224, 224]).to(device)
    cnn = ResNet18(num_classes=2).to(device) # 导入VGG19模型并且将数据转到显存中
    gpu_tracker.track()