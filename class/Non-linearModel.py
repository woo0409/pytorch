import torch
import torch.nn.functional as F
from utils.utils import plot_curve

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
y_data = torch.Tensor([[0], [0], [0], [0], [1], [1], [1]])
loss_list = []


# 新建类继承torch.nn.Module模块
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # 第一个参数是输入样本的size，输入样本是几维的
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 通过linear自动计算的预测值并返回
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


# 实例化模型对象
model = LogisticRegressionModel()
# 损失函数计算模型均值方差的对象
criterion = torch.nn.BCELoss(size_average=False)
# 实例化优化器对象，用于对模型进行参数更新
optimizer = torch.optim.SGD(model.parameters(), lr=0.025)

for epoch in range(10000):
    # 1.计算预测值
    y_pred = model(x_data)
    # 2.计算损失值
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    # 3.将模型中的参数清零
    optimizer.zero_grad()
    # 4.进行梯度的反向传播
    loss.backward()
    # 5.将模型的参数进行更新
    optimizer.step()

    loss_list.append(loss.data)


print("w= ", model.linear.weight.item())
print("b= ", model.linear.bias.item())

x_test = torch.Tensor([[2.54]])
y_test = model(x_test)
print("y_test = %.10f" % y_test.data)
plot_curve(loss_list)
