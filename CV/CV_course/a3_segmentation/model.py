import torch.nn as nn
import torch.nn.functional as F


class LeNet_seg(nn.Module):
    def __init__(self):
        super(LeNet_seg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)  # 没padding
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(p=0.05)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=1, padding=0, output_padding=0, bias=True)
        self.deconv2 = nn.ConvTranspose2d(6, 11, 5, padding=2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.cls = nn.Linear(11, 11)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.relu(self.conv1(x))  # [b,6,28,28]
        x, pool_idx1 = F.max_pool2d(x, 2, 2, return_indices=True)  # [b,6,14,14]
        # x = F.relu(self.conv2(x))  # [b,16,10,10]
        # x, pool_idx2 = F.max_pool2d(x, 2, 2, return_indices=True)  # [b,16,5,5]
        # # cls
        # x = x.view(b, -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # seg
        # x_ = self.unpool1(x, pool_idx2)  # [b,16,10,10]
        # x_ = self.deconv1(x)  # [b,6,14,14]
        x_ = self.unpool1(x, pool_idx1)  # [b,6,28,28]
        x_ = self.deconv2(x_)  # [b,11,28,28]
        x_ = x_.permute(0, 2, 3, 1)
        # x_ = self.cls(x_))
        return F.log_softmax(x, dim=-1), F.log_softmax(x_, dim=-1)  # , F.sigmoid(x_)


class LeNet_seg_simple(nn.Module):
    def __init__(self):
        super(LeNet_seg_simple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=11, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        b, _, h, w = x.shape
        x_ = self.conv1(x)  # [b,6,28,28]
        x_ = x_.permute(0, 2, 3, 1)

        return F.log_softmax(x, dim=-1), F.log_softmax(x_, dim=-1)  # , F.sigmoid(x_)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)  # 没padding
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(p=0.05)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.relu(self.conv1(x))  # [b,6,28,28]
        x = F.max_pool2d(x, 2, 2)  # [b,6,14,14]
        x = self.drop(x)
        # print(x.shape)
        # exit()
        x = F.relu(self.conv2(x))  # [b,16,10,10]
        x = F.max_pool2d(x, 2, 2)  # [b,16,5,5]
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LeNet_half(nn.Module):
    def __init__(self):
        super(LeNet_half, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(3, 8, 5, 1)  # 没padding
        self.fc1 = nn.Linear(5 * 5 * 8, 60)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, 10)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.relu(self.conv1(x))  # [b,6,28,28]
        x = F.max_pool2d(x, 2, 2)  # [b,6,14,14]
        x = F.relu(self.conv2(x))  # [b,16,10,10]
        x = F.max_pool2d(x, 2, 2)  # [b,16,5,5]
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MLP_seg(nn.Module):
    def __init__(self):
        super(MLP_seg, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1000)  # for cls
        self.cls = nn.Linear(1000, 11 * 28 * 28)

    def forward(self, x):
        b, _, h, w = x.size()
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = self.cls(x)
        x = x.view(b, h, w, 11)
        return None, F.log_softmax(x, -1)


class MLP_seg_two_stage(nn.Module):
    def __init__(self):
        super(MLP_seg_two_stage, self).__init__()
        self.fc1_ = nn.Linear(28 * 28, 1024)  # for cls
        self.cls = nn.Linear(1024, 10)

        self.fc1 = nn.Linear(28 * 28, 1000)
        self.fc2 = nn.Linear(1000, 28 * 28)

    def forward(self, x):
        ori_shape = x.size()
        x = x.view(x.size(0), -1)
        x_ = F.relu(self.fc1_(x))
        cls = self.cls(x_)

        x = F.tanh(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = x.view(ori_shape)
        return F.log_softmax(cls, 1), x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class MLP_2(nn.Module):
    def __init__(self):
        super(MLP_2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLP_3(nn.Module):
    def __init__(self):
        super(MLP_3, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    import numpy as np

    writer = SummaryWriter('./tensorboard')
    writer.add_histogram('normal_centered', np.random.normal(0, 1, 1000), global_step=1)
    writer.add_histogram('normal_centered', np.random.normal(0, 2, 1000), global_step=50)
    writer.add_histogram('normal_centered', np.random.normal(0, 3, 1000), global_step=100)
    # for i in range(1000):
    #     writer.add_scalar('quadratic', i ** 2, global_step=i)
    # writer.add_scalar('exponential', 2 ** i, global_step=i)
