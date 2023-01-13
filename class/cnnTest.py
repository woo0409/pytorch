import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv

batch_size = 32
img_path = "../Pic/img.jpg"
# 将图像灰度值转为0-1之间的数值
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 得到数据集
train_dataset = datasets.MNIST(root='mnist_data',
                               train=True,
                               transform=transform,
                               download=False)

test_dataset = datasets.MNIST(root='mnist_data/',
                              train=False,
                              download=False,
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义CNN神经网络
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 20, kernel_size=5, padding=2)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(980, 512)
        # self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(512, 128)
        # self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(128, 32)
        self.fc6 = torch.nn.Linear(32, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)
        # return x


# 模型初始化和定义损失函数和反向传播算法
model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.5)

# test_data = np.full(shape=[1, 1, 28, 28], fill_value=0.5, dtype=np.float32)
# test_data = torch.from_numpy(test_data)
# print(test_data.shape)
# print(test_data)
# rus = model(test_data)
# print(rus.shape)

# 训练模型
def train(epoch):
    running_loss = 0.0
    # 在train_loader中得到训练数据
    for idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # 1. 正向传播
        output = model(inputs)
        # 2. 反向传播（计算loss函数并反馈）, 传播前先清零梯度值
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()
        # 3. 更新参数
        optimizer.step()

        running_loss += loss.item()

        if idx % 300 ==299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 300))
            running_loss = 0.0


# 测试部分
def test():
    # 需要计算正确率就必须统计总个数和答对的个数
    correct = 0
    total_number = 0
    # 测试部分就不需要计算梯度
    with torch.no_grad():
        for data in test_loader:
            image_input, image_label = data
            image_output = model(image_input)
            # 取每一行（dim=1表第一个维度）最大值（max）的下标(predicted)及最大值(_)
            _, predicted = torch.max(image_output.data, dim=1)

            # total_number += 1
            total_number += image_label.size(0)
            correct += (predicted == image_label).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total_number))


if __name__ == '__main__':
    img = cv.imread(img_path)
    tensor_cv = torch.from_numpy(np.transpose(img, (2, 0, 1)))[0].reshape([1, 1, 28, 28])
    tensor_cv = tensor_cv.float()
    # print(tensor_cv)

    for epoch in range(5):
        train(epoch)
        test()

        result = model(tensor_cv)
        _, pred = torch.max(result.data, dim=1)
        print("pred: ", pred)
