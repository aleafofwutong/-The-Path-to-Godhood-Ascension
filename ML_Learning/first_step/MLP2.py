import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------
# 1. 激活函数及导数（核心组件）
# --------------------------
def relu(x):
    """ReLU 激活函数：f(x) = max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 导数：x>0 时为 1，否则为 0"""
    return np.where(x > 0, 1, 0)

def softmax(x):
    """Softmax 激活函数：将输出转为概率分布（分类任务输出层）"""
    # 减去最大值避免指数溢出
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --------------------------
# 2. MLP 类定义（核心逻辑）
# --------------------------
class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=0.01):
        """
        初始化 MLP
        :param input_dim: 输入特征维度（如 2 维数据）
        :param hidden_dims: 隐藏层维度列表（如 [16, 8] 表示 2 个隐藏层，分别 16/8 个神经元）
        :param output_dim: 输出维度（分类任务为类别数，回归任务为 1）
        :param learning_rate: 学习率
        """
        self.lr = learning_rate
        self.layers = []  # 存储各层权重和偏置（(W, b)）
        
        # 初始化输入层 -> 第一个隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # 权重 W：(前一层维度, 当前层维度)，用 Xavier 初始化（避免梯度消失/爆炸）
            W = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2 / (prev_dim + hidden_dim))
            b = np.zeros((1, hidden_dim))  # 偏置 b：(1, 当前层维度)，初始为 0
            self.layers.append((W, b))
            prev_dim = hidden_dim
        
        # 初始化最后一个隐藏层 -> 输出层
        W_out = np.random.randn(prev_dim, output_dim) * np.sqrt(2 / (prev_dim + output_dim))
        b_out = np.zeros((1, output_dim))
        self.layers.append((W_out, b_out))
    
    def forward(self, X):
        """前向传播：计算输入 X 的输出（返回各层激活值，用于反向传播）"""
        activations = [X]  # 存储各层激活值（输入层为原始 X）
        current = X
        
        # 隐藏层：ReLU 激活
        for i in range(len(self.layers) - 1):
            W, b = self.layers[i]
            z = np.dot(current, W) + b  # 线性变换：z = X@W + b
            current = relu(z)  # 激活函数
            activations.append(current)
        
        # 输出层：Softmax 激活（分类任务）
        W_out, b_out = self.layers[-1]
        z_out = np.dot(current, W_out) + b_out
        output = softmax(z_out)
        activations.append(output)
        
        return activations  # activations[0] = X, activations[-1] = 输出
    
    def compute_loss(self, y_pred, y_true):
        """计算交叉熵损失（分类任务）"""
        # y_true 需为 one-hot 编码（如 [0,1,0] 表示第二类）
        n_samples = y_true.shape[0]
        # 避免 log(0) 溢出，加微小值
        loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / n_samples
        return loss
    
    def backward(self, activations, y_true):
        """反向传播：计算梯度并更新权重/偏置"""
        n_samples = y_true.shape[0]
        gradients = []  # 存储各层梯度（dW, db）
        
        # 1. 输出层梯度计算
        output = activations[-1]
        # 交叉熵 + Softmax 的导数简化：dL/dz_out = y_pred - y_true
        delta = output - y_true  # (n_samples, output_dim)
        
        # 输出层权重和偏置的梯度
        W_out, b_out = self.layers[-1]
        prev_activation = activations[-2]
        dW_out = np.dot(prev_activation.T, delta) / n_samples  # (prev_dim, output_dim)
        db_out = np.sum(delta, axis=0, keepdims=True) / n_samples  # (1, output_dim)
        gradients.append((dW_out, db_out))
        
        # 2. 隐藏层梯度计算（从后往前）
        for i in reversed(range(len(self.layers) - 1)):
            W_curr, b_curr = self.layers[i]
            W_next = self.layers[i + 1][0]  # 下一层的权重
            prev_activation = activations[i]  # 当前层的输入激活值
            curr_activation = activations[i + 1]  # 当前层的输出激活值
            
            # 梯度反向传播：delta = (delta_next @ W_next.T) * relu'(z_curr)
            delta = np.dot(delta, W_next.T) * relu_derivative(curr_activation)
            # 当前层权重和偏置的梯度
            dW = np.dot(prev_activation.T, delta) / n_samples
            db = np.sum(delta, axis=0, keepdims=True) / n_samples
            gradients.append((dW, db))
        
        # 3. 反向更新权重和偏置（ gradients 是反向存储的，需反转）
        gradients = gradients[::-1]
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            dW, db = gradients[i]
            self.layers[i] = (W - self.lr * dW, b - self.lr * db)
    
    def train(self, X_train, y_train, epochs=1000, batch_size=32, verbose=True):
        """训练模型"""
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            # 随机打乱数据（批量梯度下降）
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_loss = 0
            # 批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                activations = self.forward(X_batch)
                y_pred = activations[-1]
                # 计算损失
                loss = self.compute_loss(y_pred, y_batch)
                total_loss += loss * len(X_batch)
                # 反向传播 + 更新参数
                self.backward(activations, y_batch)
            
            # 计算 epoch 平均损失
            avg_loss = total_loss / n_samples
            if verbose and (epoch + 1) % 100 == 0:
                # 计算训练准确率
                y_pred_train = self.predict(X_train)
                acc = np.mean(np.argmax(y_train, axis=1) == y_pred_train)
                print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")
    
    def predict(self, X):
        """预测：返回类别索引（分类任务）"""
        activations = self.forward(X)
        y_pred = activations[-1]
        return np.argmax(y_pred, axis=1)

# --------------------------
# 3. 数据准备（二分类任务示例）
# --------------------------
# 生成非线性可分数据（月亮数据集）
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 特征标准化（提升训练效果）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 标签 one-hot 编码（适应分类任务输出）
y_train_onehot = np.eye(2)[y_train]  # (800, 2)
y_test_onehot = np.eye(2)[y_test]    # (200, 2)

# --------------------------
# 4. 模型训练与评估
# --------------------------
# 初始化 MLP：输入维度=2，隐藏层=[16, 8]，输出维度=2（2分类）
mlp = MLP(
    input_dim=2,
    hidden_dims=[16, 8],  # 2个隐藏层，16和8个神经元
    output_dim=2,
    learning_rate=0.05
)

# 训练模型
print("开始训练 MLP...")
mlp.train(X_train, y_train_onehot, epochs=2000, batch_size=64)

# 测试集评估
y_pred_test = mlp.predict(X_test)
test_acc = np.mean(y_pred_test == y_test)
print(f"\n测试集准确率：{test_acc:.4f}")

# --------------------------
# 5. 可视化决策边界（可选）
# --------------------------
def plot_decision_boundary(model, X, y):
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('MLP Decision Boundary')
    plt.show()

plot_decision_boundary(mlp, X_test, y_test)