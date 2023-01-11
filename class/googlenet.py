import torch
# 组建DataLoader
from torchvision import transforms  # 图像
from torchvision import datasets
from torch.utils.data import DataLoader
# 激活函数和优化器
import torch.nn.functional as F
import torch.optim as optim

# Dataset&Dataloader必备
batch_size = 64
# pillow（PIL）读的原图像格式为W*H*C，原值较大
# 转为格式为C*W*H值为0-1的Tensor
transform = transforms.Compose([
    # 变为格式为C*W*H的Tensor
    transforms.ToTensor(),
    # 第一个是均值，第二个是标准差，变值为0-1
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='mnist_data',
                               train=True,
                               download=False,
                               transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='mnist_data/',
                              train=False,
                              download=False,
                              transform=transform)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class InceptionA(torch.nn.Module):
    # 仅是一个模块，其中的输入通道数并不能够指明
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch5x5_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


# 定义神经网络结构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # 在Inception的定义中，拼接后的输出通道数为24+16+24+24=88个
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        # 关于1408：
        # 每次卷积核是5x5，则卷积后原28x28的图像变为24x24的
        # 再经过最大池化，变为12x12的
        # 以此类推最终得到4x4的图像，又inception输出通道88，则转为一维后为88x4x4=1408个
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)

        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)

        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


model = Net()
print("GPU使用情况", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
# 随机梯度下降，momentum表冲量，在更新时一定程度上保留原方向
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    # 提取数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        # 优化器清零
        optimizer.zero_grad()
        # 前馈
        outputs = model(inputs)
        # 反馈
        loss = criterion(outputs, target)
        loss.backward()
        # 参数更新
        optimizer.step()
        # 累计loss
        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    # 避免计算梯度
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取每一行（dim=1表第一个维度）最大值（max）的下标(predicted)及最大值(_)
            _, predicted = torch.max(outputs.data, dim=1)
            # 加上这一个批量的总数（batch_size），label的形式为[N,1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
