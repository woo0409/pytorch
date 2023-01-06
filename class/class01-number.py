import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch import optim
from matplotlib import pyplot as plt
from utils.utils import plot_image, plot_curve ,one_hot


batch_size = 512

# 加载数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


class Net(nn.Module):

    # 搭建神经网络结构的过程
    def __init__(self):
        super(Net, self).__init__()

        # 搭建三层神经网络
        # 第一个参数指图像的像素个数，即第一层神经网络的输入，第二个参数为第二层网络的神经节点，可有自己决定
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        # 最后一层的第二个参数是输出层神经节点个数，一定是结果类别的个数
        self.fc3 = nn.Linear(64, 10)

    # 进行前向传播过程
    def forward(self, x):
        # 通过第一层计算出第二层神经网路的激活值并使用relu函数处理
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []

for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将x数据将为二维,view函数相当于reshape
        x = x.view(x.size(0), 28 * 28)
        # print(x)
        # net传入的参数形式为[x, label]
        out = net(x)
        # 将y转换为独热码
        y_onehot = one_hot(y)
        # 通过计算loss函数值来确定预测值的质量
        loss = F.mse_loss(out, y_onehot)

        # 根据计算出的loss值向后传播进行参数的更新
        # 清零梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()

        train_loss.append(loss.item())
# 完成训练，得到合适的神经网络参数
plot_curve(train_loss)


# 测试部分
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)

    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_number = len(test_loader.dataset)
acc = total_correct / total_number
print("test acc :", acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, "test")