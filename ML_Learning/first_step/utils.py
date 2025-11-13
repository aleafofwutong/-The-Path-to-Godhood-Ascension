import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict

class Linear_layer(nn.Module):
    """自定义线性层（封装nn.Linear，添加simulate手动模拟）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        return self.linear(x)

    def simulate(self, x):
        """手动模拟线性变换：y = x @ weight.T + bias,和forward数学逻辑一致"""
        # x: (batch, in_channels) → weight: (out_channels, in_channels) → 矩阵乘法需转置
        return x @ self.weight.T + self.bias


class Flatten_Layer(nn.Module):
    """展平层（使用官方nn.Flatten，添加simulate手动模拟）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Flatten()  # 训练用：官方展平层

    def forward(self, x):
        # 训练用：官方实现
        return self.net(x)

    def simulate(self, x):
        """手动模拟展平：(batch, C, H, W) → (batch, C*H*W)"""
        return x.view(x.shape[0], -1)


class Softmax_Layer(nn.Module):
    """Softmax层（支持官方实现和手动simulate）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Softmax(dim=self.dim)  # 训练用：官方Softmax

    def forward(self, x):
        # 训练用：官方实现
        return self.net(x)

    def simulate(self, x):
        """手动模拟Softmax：exp(x) / sum(exp(x), dim=dim)"""
        x_exp = x.exp()
        # 加keepdim=True避免广播维度不匹配
        partition = x_exp.sum(dim=self.dim, keepdim=True)
        return x_exp / partition


class A_simple_net(nn.Module):
    """精简版Softmax回归模型（添加simulate，手动模拟整个网络计算）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ('flatten', Flatten_Layer()),
                ('linear', Linear_layer(2*28*28, 10))
            ])
        )
        # 初始化参数（训练和simulate共用同一套参数）
        init.normal_(self.net.linear.linear.weight, mean=0, std=0.01)
        init.constant_(self.net.linear.linear.bias, val=0.)
        # 暴露各层，方便simulate调用
        self.flatten = self.net.flatten
        self.linear = self.net.linear

    def forward(self, x):
        # 训练用：官方层串联（高效）
        return self.net(x)

    def simulate(self, x):
        """手动模拟整个网络流程：flatten.simulate → linear.simulate"""
        x_flat = self.flatten.simulate(x)
        x_linear = self.linear.simulate(x_flat)
        return x_linear  # 输出logits（无Softmax）


class Softmax_Regression(nn.Module):
    """带Softmax输出的模型（添加simulate，手动模拟整个网络计算）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ("flatten", Flatten_Layer()),
                ("linear", Linear_layer(2*28*28, 10)),
                ("softmax", Softmax_Layer(dim=1))
            ])
        )
        # 暴露各层，方便simulate调用
        self.flatten = self.net.flatten
        self.linear = self.net.linear
        self.softmax = self.net.softmax

    def forward(self, x):
        return self.net(x)

    def simulate(self, x):
        x_flat = self.flatten.simulate(x)
        x_linear = self.linear.simulate(x_flat)
        x_softmax = self.softmax.simulate(x_linear)
        return x_softmax


# ------------------------------ 损失函数（添加simulate，手动模拟交叉熵计算）------------------------------
class Loss_cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.CrossEntropyLoss()  # 训练用：官方损失

    def forward(self, x, y):
        # 训练用：官方实现（高效稳定，内置Softmax）
        return self.net(x, y)

    def simulate(self, x, y):
        """手动模拟交叉熵损失（适用于A_simple_net的logits输出）"""
        # 步骤1：对logits做Softmax得到概率
        x_softmax = torch.softmax(x, dim=1)
        # 步骤2：取真实标签对应的概率（y为类别索引）
        batch_size = x.shape[0]
        # 构造one-hot编码（和y对应）
        y_one_hot = torch.zeros_like(x_softmax)
        y_one_hot.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        # 步骤3：计算交叉熵（平均损失）
        cross_entropy = -torch.sum(y_one_hot * torch.log(x_softmax + 1e-10)) / batch_size  # +1e-10避免log(0)
        return cross_entropy


# ------------------------------ 测试代码（验证forward和simulate结果一致性）------------------------------
if __name__ == '__main__':
    # 构造测试输入：(batch=3, channels=2, height=28, width=28)
    x = torch.randn(3, 2, 28, 28)
    # 构造测试标签：(batch=3) 类别索引（0-9）
    y = torch.randint(0, 10, (3,))

    # ------------------------------ 测试A_simple_net ------------------------------
    print("=== 测试A_simple_net（forward vs simulate）===")
    net1 = A_simple_net()
    # 训练用forward输出（logits）
    out_forward1 = net1(x)
    # 手动simulate输出（logits）
    out_simulate1 = net1.simulate(x)
    # 验证结果一致性（误差应接近0）
    print(f"forward输出形状: {out_forward1.shape}")
    print(f"simulate输出形状: {out_simulate1.shape}")
    print(f"logits误差（L2范数）: {torch.norm(out_forward1 - out_simulate1).item():.6f}")

    # ------------------------------ 测试Softmax_Regression ------------------------------
    print("\n=== 测试Softmax_Regression（forward vs simulate）===")
    net2 = Softmax_Regression()
    out_forward2 = net2(x)
    out_simulate2 = net2.simulate(x)
    print(f"forward输出形状: {out_forward2.shape}")
    print(f"simulate输出形状: {out_simulate2.shape}")
    print(f"概率误差（L2范数）: {torch.norm(out_forward2 - out_simulate2).item():.6f}")
    print(f"simulate概率和: {out_simulate2.sum(dim=1).detach().numpy()}")

    # ------------------------------ 测试Loss_cross_entropy ------------------------------
    print("\n=== 测试Loss_cross_entropy（forward vs simulate）===")
    loss_func = Loss_cross_entropy()
    # 用A_simple_net的logits计算损失
    loss_forward = loss_func(out_forward1, y)
    loss_simulate = loss_func.simulate(out_forward1, y)
    print(f"forward损失值: {loss_forward.item():.6f}")
    print(f"simulate损失值: {loss_simulate.item():.6f}")
    print(f"损失误差: {abs(loss_forward.item() - loss_simulate.item()):.6f}")