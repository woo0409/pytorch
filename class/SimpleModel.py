import torch
from utils.utils import plot_curve

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
loss_list = []


# 新建类继承torch.nn.Module模块
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 第一个参数是输入样本的size，输入样本是几维的
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 通过linear自动计算的预测值并返回
        y_pred = self.linear(x)
        return y_pred


# 实例化模型对象
model = LinearModel()
# 损失函数计算模型均值方差的对象
criterion = torch.nn.MSELoss(size_average=True)
# 实例化优化器对象，用于对模型进行参数更新
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
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

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_test =", y_test.data)
plot_curve(loss_list)
