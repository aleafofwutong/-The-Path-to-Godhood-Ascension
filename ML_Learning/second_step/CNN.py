import torch
import torch.nn as nn
from pathlib import Path
import sys
import torchvision

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 输入1通道（MNIST灰度图），输出32通道
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 输入32通道，输出64通道
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个类别（0-9）
        self.inspection = nn.InstanceNorm2d(32)  # 对应conv1的32通道
        self.inspec2 = nn.InstanceNorm2d(64)     # 对应conv2的64通道

    def forward(self, x):
        x = self.conv1(x)
        x = self.inspection(x)  # 修正：用32通道的InstanceNorm
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.inspec2(x)     # 正确：用64通道的InstanceNorm
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)  # 2x2池化，输出12x12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # 展平为(batch_size, 64*12*12=9216)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x  # 直接输出fc2的logits
        return output

# 初始化模型和设备
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 修正：用CrossEntropyLoss（更稳定，适配logits输出）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 加载训练集（正确）
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准归一化参数（正确）
])
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(  # 新增：加载测试集（train=False）
    root='./data', train=False, download=True, transform=transform
)

# 训练模型
model.train()
epochs = 10
for epoch in range(1, epochs + 1):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # 输出logits
        loss = criterion(output, target)  # CrossEntropyLoss直接接收logits和target
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:  # 每100个batch打印一次（避免输出过多）
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 保存模型（正确）
torch.save(model.state_dict(), 'mnist_cnn.pt')

# 加载模型并测试
model2 = CNN()
model2.to(device)
state_dict = torch.load("mnist_cnn.pt", map_location=device)  # 加载时指定设备，避免不匹配
model2.load_state_dict(state_dict)
model2.eval()  # 关键：进入评估模式，关闭Dropout

# 测试：用测试集（正确）
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
correct = 0
total = 0
with torch.no_grad():  # 测试时禁用梯度计算，节省内存和时间
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model2(data)
        pred = output.argmax(dim=1, keepdim=True)  # 取概率最大的类别
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

# 计算并打印测试集准确率（正确：统计所有测试样本）
accuracy = 100. * correct / total
print('\nTest Accuracy: {}/{} ({:.2f}%)'.format(correct, total, accuracy))