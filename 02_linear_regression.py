import torch
import torch.nn as nn

# 1. 数据
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])

y = torch.tensor([[3.0],
                  [5.0],
                  [7.0],
                  [9.0]])

# 2. 模型
model = nn.Linear(1, 1)

# 3. 损失函数
loss_fn = nn.MSELoss()

# 4. 优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# 5. 训练
for epoch in range(1000):

    y_pred = model(x)

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss = {loss.item():.6f}, "
            f"w = {model.weight.item():.4f}, "
            f"b = {model.bias.item():.4f}"
        )

# 6. 最终参数
print("\n最终结果：")
print("w =", model.weight.item())
print("b =", model.bias.item())

# 7. 测试
x_test = torch.tensor([[10.0]])

with torch.no_grad():
    y_test = model(x_test)

print("当 x = 10 时，模型预测 y =", y_test.item())