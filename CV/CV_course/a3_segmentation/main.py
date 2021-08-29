import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from a2_mnist.config import get_args
from a2_mnist.dataset import get_mnist_dataset
from a2_mnist.model import LeNet, MLP, MLP_2, MLP_3, LeNet_half, MLP_seg, LeNet_seg, LeNet_seg_simple
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter('./tensorboard')


def show_img(img):
    img = np.array(img) * 255
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('tmp.jpg', img)
    plt.imshow(img)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # writer.add_graph(model, input_to_model=torch.zeros(64, 1, 28, 28).cuda(), verbose=False)
    for batch_idx, (data, target) in enumerate(train_loader):
        show_img(data[0][0].cpu().numpy())
        exit()
        # print(data.mean(), data.std())
        # print(data[0])
        b = data.size(0)
        data, target = data.to(device), target.to(device)
        pixel_label, bg_mask = torch.zeros(data.size()).cuda().long(), torch.zeros(data.size()).cuda().long()
        pixel_label[data > 0] = 1
        pixel_label = pixel_label * (target.view(b, 1, 1, 1) + 1)
        bg_mask[data == 0] = 1
        data = (data - 0.1307) / 0.3081
        noise = torch.randn(data.size()).cuda() * 1
        noise *= bg_mask
        data += noise

        optimizer.zero_grad()
        cls, pixel = model(data)
        # loss = F.nll_loss(cls, target)
        # pixel loss
        # print(pixel_label.shape)
        loss = F.nll_loss(pixel.view(-1, 11), pixel_label.view(-1))
        # print(pixel[0])
        # loss -= (torch.log(pixel) * pixel_label).mean() * 10

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # acc = test(args, model, device, test_loader)
            # writer.add_histogram('loss_LeNet', loss.item(), batch_idx)
            # writer.add_histogram('acc_LeNet', acc, batch_idx)  # add_scalar


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pixel_rate = 0
    pixel_binary_rate = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            b, _, h, w = data.size()
            pixel_label, bg_mask = torch.zeros(data.size()).cuda().long(), torch.zeros(data.size()).cuda().long()
            pixel_label[data > 0] = 1
            pixel_label = (pixel_label * (target.view(b, 1, 1, 1) + 1)).squeeze(1)
            bg_mask[data == 0] = 1
            data = (data - 0.1307) / 0.3081
            noise = torch.randn(data.size()).cuda() * 0.3
            noise *= bg_mask
            data += noise
            output, pixel = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

            # TODO:pixel acc
            count = torch.zeros(pixel_label.size()).long()
            pred = pixel.argmax(dim=-1, keepdim=False)
            count[pred == pixel_label] = 1
            # print(pred[0])
            count_bi = torch.zeros(pixel_label.size()).long()
            tmp1 = torch.zeros(pixel_label.size()).long()
            tmp2 = torch.zeros(pixel_label.size()).long()
            tmp1[pred > 0] = 1
            tmp2[pixel_label > 0] = 1
            count_bi[tmp1 == tmp2] = 1

            pixel_rate += count.sum().item()
            pixel_binary_rate += count_bi.sum().item()
            # print(pixel_rate)
            # thre = 0.99999
            # # print(pixel[0])
            # pixel[pixel > thre] = 1
            # pixel[pixel < thre] = 0
            # # exit()
            # # print((1 - (abs(pixel - pixel_label).sum() / (b * h * w))))
            # pixel_rate += abs(pixel - pixel_label).sum()  # 所有错误的像素点
            # print(pixel_rate)

    test_loss /= len(test_loader.dataset)
    pixel_rate /= (len(test_loader.dataset) * 28 * 28)
    pixel_binary_rate /= (len(test_loader.dataset) * 28 * 28)

    print(pixel_binary_rate, pixel_rate)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


if __name__ == '__main__':
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = get_mnist_dataset(args, kwargs)

    model = LeNet_seg().to(device)
    # model = MLP_seg().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
