from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from pathlib import Path
from tensorboardX import SummaryWriter
from datetime import datetime
import os

from dataPrepare import MovieLens
from fm import FM
from utils.utils import mkdirs


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training on [{}].'.format(device))

    # 定义超参
    lr_init = 0.001
    max_epoch = 4
    train_bn = 32
    test_bn = 32

    # 数据目录
    data_path = Path(__file__).absolute().parents[0]
    # log
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = data_path / 'data' / time_str / 'log'
    mkdirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # 加载数据
    root_path = Path(__file__).absolute().parents[2]
    ## print(root_path)
    train_dataset_path = root_path / 'data/ml-1m/rating_user_movie_merge_train.csv'
    test_dataset_path = root_path / 'data/ml-1m/rating_user_movie_merge_test.csv'

    train_data = MovieLens(train_dataset_path)
    test_data = MovieLens(test_dataset_path)

    train_loader = DataLoader(dataset=train_data, batch_size=train_bn, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_bn)

    # 定义网络
    field_dims = [6041, 3953, 2, 7, 21, 18]
    net = FM(field_dims, 5)
    net.train()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # nn.CrossEntropyLoss()  # 选择损失函数
    # 选择优化器
    optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
    # 设置学习率下降策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    # 训练
    for epoch in range(max_epoch):
        loss_sigma = 0.0  # 记录一个epoch的loss之和
        correct = 0.0  # 正确的数量
        total = 0.0  # 总共的数量

        for i, data in enumerate(train_loader):
            # 获取数据
            inputs, labels = data
            labels = Variable(labels)
            inputs = Variable(inputs)

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = net(inputs)
            # outputs = torch.tensor(outputs, dtype=torch.float, requires_grad=True)
            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()  # loss反向传播
            optimizer.step()  # 更新参数

            # 统计预测信息
            total += labels.size(0)
            outputs = torch.tensor([1 if c > 0.6 else 0 for c in outputs], dtype=torch.float,
                                   requires_grad=True).reshape(-1, 1)
            correct += (outputs.squeeze(1) == labels).squeeze().sum().numpy()
            loss_sigma += loss.item()

            # 每10个iteration，打印一次信息
            print_cycle = 100
            if i % (print_cycle - 1) == 0:
                loss_avg = loss_sigma / print_cycle
                loss_sigma = 0.0
                print(
                    f"Training: Epoch[{epoch + 1}/{max_epoch}] Iteration[{i + 1}/{len(train_loader)}] Loss: {loss_avg} Acc:{correct / total}")

                # 记录训练loss
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # 记录learning rate
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
                # 记录Accuracy
                writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

        # 更新学习率
        if epoch % 2 == 0:
            scheduler.step()

            # 每个epoch，记录梯度，权值
        for name, layer in net.named_parameters():
            writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        # 观察模型在验证集上的表现
        if epoch % 2 == 0:
            loss_sigma = 0.0
            net.eval()
            for i, data in enumerate(test_loader):
                inputs, labels = data
                labels = Variable(labels)
                inputs = Variable(inputs)
                # forward
                outputs = net(inputs)
                outputs.detach()
                # 计算损失
                loss = criterion(outputs.squeeze(1), labels.float())
                loss_sigma += loss.item()
                # 计算精度
                total += labels.size(0)
                correct += (outputs.squeeze(1) == labels).squeeze().sum().numpy()

            print(f'Testing: Valid-Accuracy:{correct / total} Valid-Loss:{loss_sigma / len(test_loader)}')
            # 记录Loss, accuracy
            writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(test_loader)}, epoch)
            writer.add_scalars('Accuracy_group', {'valid_acc': correct / total}, epoch)

    # 保存模型
    model_save_dir = data_path / 'data' / time_str / 'last_model_file'
    mkdirs(model_save_dir)
    torch.save(net.state_dict(), os.path.join(str(model_save_dir), 'net_params.pkl'))
    print('=============FINISH==========================')


if __name__ == '__main__':
    main()
