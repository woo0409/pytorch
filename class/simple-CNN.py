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


# 定义神经网络结构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(320, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
