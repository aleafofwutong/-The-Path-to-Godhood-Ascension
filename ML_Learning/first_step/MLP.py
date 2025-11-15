import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------- 1. 配置参数（可按需修改）--------------------------
batch_size = 64  # 每次训练的样本数
learning_rate = 0.001  # 学习率
epochs = 10  # 训练轮数
input_dim = 784  # 输入维度（MNIST图片28×28=784）
hidden_dim1 = 256  # 第一层隐藏层神经元数
hidden_dim2 = 128  # 第二层隐藏层神经元数
output_dim = 10  # 输出维度（10个数字分类）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU/CPU
# -------------------------- 2. 数据准备（MNIST数据集）--------------------------
# 数据预处理：转换为Tensor + 归一化（0-1区间）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为[0,1]的Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差（优化训练）
])

# 下载/加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 数据加载器（批量加载+打乱数据）
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 可视化一个样本（可选，帮助理解数据）
def visualize_sample():
    images, labels = next(iter(train_loader))
    image = images[0].numpy().squeeze()  # 转换为28×28的numpy数组
    label = labels[0].item()
    plt.imshow(image, cmap='gray')
    plt.title(f"Sample Label: {label}")
    plt.show()

visualize_sample()  # 运行后会显示一张手写数字图片

# -------------------------- 3. 定义MLP模型 --------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        # 定义网络层
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),  # 输入层→第一层隐藏层
            nn.ReLU(),  # 激活函数（引入非线性）
            nn.Dropout(0.2),  # Dropout层（防止过拟合，随机丢弃20%神经元）
            nn.Linear(hidden_dim1, hidden_dim2),  # 第一层隐藏层→第二层隐藏层
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.2),  # 再次Dropout
            nn.Linear(hidden_dim2, output_dim)  # 第二层隐藏层→输出层
        )

    def forward(self, x):
        # 前向传播（数据流经网络的过程）
        x = x.view(-1, input_dim)  # 展平输入（batch_size, 28×28）→ (batch_size, 784)
        output = self.layers(x)
        return output

# 初始化模型并移到GPU/CPU
model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
print("MLP模型结构：")
print(model)  # 打印模型结构

# -------------------------- 4. 定义损失函数和优化器 --------------------------
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适合分类任务）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器（常用且稳定）

# -------------------------- 5. 训练模型 --------------------------
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 切换到训练模式（启用Dropout）
    total_loss = 0.0
    correct = 0  # 记录正确分类的样本数

    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移到GPU/CPU
        data, target = data.to(device), target.to(device)

        # 前向传播：计算模型输出
        output = model(data)
        loss = criterion(output, target)  # 计算损失

        # 反向传播：更新参数
        optimizer.zero_grad()  # 清空上一轮梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新权重

        # 统计信息
        total_loss += loss.item() * data.size(0)  # 累计损失
        pred = output.argmax(dim=1, keepdim=True)  # 预测类别（概率最大的类别）
        correct += pred.eq(target.view_as(pred)).sum().item()  # 累计正确数

    # 计算本轮平均损失和准确率
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# -------------------------- 6. 测试模型 --------------------------
def test(model, test_loader, criterion):
    model.eval()  # 切换到测试模式（禁用Dropout）
    total_loss = 0.0
    correct = 0

    with torch.no_grad():  # 测试时不计算梯度（节省内存+加速）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n')
    return avg_loss, accuracy

# -------------------------- 7. 运行训练和测试 --------------------------
if __name__ == "__main__":
    # 记录训练过程的损失和准确率（用于后续绘图）
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # 迭代训练
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)

        # 保存数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # -------------------------- 8. 可视化训练结果 --------------------------
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 保存模型（可选）
    torch.save(model.state_dict(), 'mlp_mnist.pth')
    print("模型已保存为 mlp_mnist.pth")