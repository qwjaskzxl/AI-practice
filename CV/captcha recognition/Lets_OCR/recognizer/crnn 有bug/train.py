#coding=utf-8

import Config
import random
import os
import numpy as np
import torch
# from warpctc_pytorch import CTCLoss
import torch.backends.cudnn as cudnn
import lib.dataset
import lib.convert
import lib.utility
from torch.autograd import Variable
import Net.net as Net
import torch.optim as optim


def val(net, criterion,test_loader, max_iter=100):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(test_loader)

    i = 0
    n_correct = 0
    loss_avg = lib.utility.averager()
    max_iter = min(max_iter, len(test_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        lib.dataset.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        lib.dataset.loadData(text, t)
        lib.dataset.loadData(length, l)
        preds = net(image)

        # print(preds.shape)
        # print(preds)

        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        preds = preds.max(2)[1]
        # print(cpu_texts, text, preds)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        # print(preds)
        # view只能用在contiguous的variable上;view()函数作用是将一个多行的Tensor,拼接成一行。
        #print("==========",preds.data)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        # print(sim_preds)
        list_1 = []
        for i in cpu_texts:
            list_1.append(i.decode('utf-8','strict'))
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1

    #raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:Config.test_disp]
    #for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        #print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * Config.batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer, train_iter):
    print('Start train')

    data = train_iter.next()
    cpu_images, cpu_texts = data

    # print(cpu_images, cpu_texts)
    # print(cpu_images[0,0,:,:][0])
    batch_size = cpu_images.size(0)
    lib.dataset.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    # print(t.min())
    lib.dataset.loadData(text, t)
    lib.dataset.loadData(length, l) #不同的长度
    # print(image.shape)
    # for i,x in enumerate(image[0,0,:,:]):
    #     print(i,x)
    preds = net(image)
    # print(preds[0, 0, :])
    # print("preds.size=%s" % preds.size)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))  # preds.size(0)=w=22
    cost = criterion(preds, text, preds_size, length) / batch_size  # length= a list that contains the len of text label in a batch
    print(cost.item())

    print(preds.mean(), preds.std())
    preds = preds.max(2)[1]
    # print(cpu_texts, text, preds)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    # print(sim_preds)

    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    if not os.path.exists(Config.model_dir):
        os.mkdir(Config.model_dir)

    print("image scale: [%s,%s]\nmodel_save_path: %s\ngpu_id: %s\nbatch_size: %s" %
          (Config.img_height, Config.img_width, Config.model_dir, Config.gpu_id, Config.batch_size))

    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的

    os.environ['CUDA_VISIBLE_DEVICES'] = Config.gpu_id  # 使用指定的GPU及GPU显存 CUDA_VISIBLE_DEVICES

    cudnn.benchmark = True
    if torch.cuda.is_available() and Config.using_cuda:
        cuda = True
        print('Using cuda')
    else:
        cuda = False
        print('Using cpu mode')

    train_dataset = lib.dataset.lmdbDataset(root=Config.train_data, transform=lib.dataset.resizeNormalize((Config.img_width, Config.img_height)))
    test_dataset = lib.dataset.lmdbDataset(root=Config.test_data, transform=lib.dataset.resizeNormalize((Config.img_width, Config.img_height)))
    assert train_dataset

    # images will be resize to 32*100
    wk_collate_fn = lib.dataset.alignCollate(imgH=Config.img_height, imgW=Config.img_width)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size,
        shuffle=False, # 设置为“真”时,在每个epoch对数据打乱.（默认：False）
        num_workers=int(Config.data_worker), # 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
        collate_fn=wk_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.batch_size,
        shuffle=False, # 设置为“真”时,在每个epoch对数据打乱.（默认：False）
        num_workers=int(Config.data_worker), # 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
        collate_fn=wk_collate_fn
    )

    n_class = len(Config.alphabet) + 1  # for python3
    # n_class = len(Config.alphabet.decode('utf-8')) + 1  # for python2
    print("alphabet class num is %s" % n_class)

    converter = lib.convert.strLabelConverter(Config.alphabet)
    # converter = lib.convert.StrConverter(Config.alphabet)
    # print(converter.dict)
    # criterion = CTCLoss()

    criterion = torch.nn.CTCLoss()
    net = Net.CRNN(n_class)
    print(net)

    net.apply(lib.utility.weights_init)

    image = torch.FloatTensor(Config.batch_size, 3, Config.img_height, Config.img_width)
    # print(image[0,1,:,:].max())
    text = torch.IntTensor(Config.batch_size * 5)  # ？
    length = torch.IntTensor(Config.batch_size)

    if cuda:
        net.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    loss_avg = lib.utility.averager()

    # optimizer = optim.RMSprop(net.parameters(), lr=Config.lr)
    # optimizer = optim.Adadelta(net.parameters(), lr=Config.lr)
    optimizer = optim.Adam(net.parameters(), lr=Config.lr,
                           betas=(Config.beta1, 0.999))

    for epoch in range(Config.epoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad = True
            net.train()

            cost = trainBatch(net, criterion, optimizer, train_iter)

            loss_avg.add(cost)
            i += 1

            if i % Config.display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, Config.epoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()


            # if (i+i) % Config.test_interval == 0:
            #     val(net, criterion, test_loader)

            # do checkpointing
            if i % Config.save_interval == 0:
                torch.save(
                    net.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(Config.model_dir, epoch, i))
