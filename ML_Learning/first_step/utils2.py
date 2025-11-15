import numpy as np
from sklearn.datasets  import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
def relu(x):
    if x > 0:
        return x
    else:
        return 0.3 * x

def relu_derivative(x):
    return np.where(x > 0, 1, 0.3)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.layers = []
        self.activations = []

        for i in range(len(hidden_dims)):
            if i == 0:
                self.weights.append(np.random.randn(input_dim, hidden_dims[i]))
                self.biases.append(np.random.randn(hidden_dims[i]))
            else:
                self.weights.append(np.random.randn(hidden_dims[i-1], hidden_dims[i]))
                self.biases.append(np.random.randn(hidden_dims[i]))

        self.weights.append(np.random.randn(hidden_dims[-1], output_dim))
        self.biases.append(np.random.randn(output_dim))

    def compute_loss(self, y_pred, y_true):
        return np.mean(-y_true * np.log(y_pred))
    
    def backward(self, activations, y_true):
        m = y_true.shape[0]
        delta = activations[-1] - y_true
        d_weights = []
        d_biases = []

        for i in range(len(self.weights)-1, -1, -1):
            if i == len(self.weights)-1:
                d_weights.append(np.dot(activations[i].T, delta) / m)
                d_biases.append(np.sum(delta, axis=0) / m)
            else:
                delta = np.dot(delta, self.weights[i+1].T) * relu_derivative(activations[i])
                d_weights.append(np.dot(activations[i].T, delta) / m)
                d_biases.append(np.sum(delta, axis=0) / m)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def train(self, X_train, y_train, epochs=1000, batch_size=32, verbose=True):
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
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    def predict(self, X):
        activations = self.forward(X)
        y_pred = activations[-1]
        return np.argmax(y_pred, axis=1)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            activations.append(a)
        return activations
    
X,y=make_moons(n_samples=1000,noise=0.1,random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

y_train_onehot=np.eye(2)[y_train]

y_test_onehot=np.eye(2)[y_test]

mlp=MLP(input_dim=2,hidden_dims=[16,8],output_dim=2,learning_rate=0.05)

mlp.train(X_train,y_train_onehot,epochs=2000,batch_size=64,verbose=True)

y_pred_test=mlp.predict(X_test)

test_acc=np.mean(y_pred_test==y_test)

def plot_decision_boundary(model,X,y):
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
    plt.savefig('result.png')

plot_decision_boundary(mlp,X_test,y_test)

