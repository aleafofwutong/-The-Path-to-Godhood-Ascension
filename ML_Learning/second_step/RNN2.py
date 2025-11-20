import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import string
from datasets import load_dataset  # 加载官方数据集

# -------------------------- 1. 文本生成模型（保持不变，适配官方数据集） --------------------------
class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

# -------------------------- 2. 加载官方数据集（wikitext-2，无需手动整理） --------------------------
# 加载 wikitext-2 数据集（train/validation/test 三分集，文本为英文维基百科内容）
dataset = load_dataset("wikitext", "wikitext-2-v1")
train_text = dataset["train"]["text"]  # 训练集文本列表（每个元素是一段文本）

# 合并所有训练文本为一个长字符串，过滤无效字符（保留英文大小写、标点、空格）
valid_chars = string.ascii_letters + string.punctuation + " "
full_text = " ".join([text.strip() for text in train_text if text.strip()])  # 合并并去空行
filtered_text = [c for c in full_text if c in valid_chars]  # 过滤特殊字符
filtered_text = "".join(filtered_text)  # 转为字符串

print(f"官方数据集训练文本长度（字符数）：{len(filtered_text)}")
print(f"前 200 个字符预览：{filtered_text[:200]}")

# -------------------------- 3. 构建字符集映射（和之前逻辑一致） --------------------------
unique_chars = sorted(list(set(filtered_text)))  # 去重排序后的字符集
vocab_size = len(unique_chars)
char2idx = {c: i for i, c in enumerate(unique_chars)}
idx2char = {i: c for i, c in enumerate(unique_chars)}

print(f"\n字符集大小：{vocab_size}")
print(f"字符集示例：{''.join(unique_chars[:20])}...")

# -------------------------- 4. 官方数据集适配（构建序列样本） --------------------------
class WikiTextDataset(Dataset):
    def __init__(self, text, char2idx, seq_len=100, step=5):
        """
        text: 过滤后的长文本字符串
        seq_len: 输入序列长度（用100个字符预测第101个）
        step: 样本步长（每隔5个字符取一个样本，平衡样本数和多样性）
        """
        self.seq_len = seq_len
        self.text_idx = [char2idx[c] for c in text]  # 文本→索引序列
        self.data = []
        
        # 生成样本（输入序列→目标序列）
        for i in range(0, len(self.text_idx) - seq_len, step):
            input_seq = self.text_idx[i:i+seq_len]
            target_seq = self.text_idx[i+1:i+seq_len+1]
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        # one-hot 编码输入序列
        input_tensor = torch.nn.functional.one_hot(
            torch.tensor(input_seq, dtype=torch.long),
            num_classes=vocab_size
        ).float()
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        return input_tensor, target_tensor

# 数据集参数（官方数据量大，可适当调大 step 减少样本数）
seq_len = 100
step = 5  # 步长5→样本数=（文本长度-100）/5，避免训练过慢
batch_size = 64  # 官方数据量大，可增大 batch_size 提升训练效率

# 创建数据集和数据加载器
wiki_dataset = WikiTextDataset(filtered_text, char2idx, seq_len=seq_len, step=step)
batch_size = min(batch_size, len(wiki_dataset)) if len(wiki_dataset) > 0 else 1
dataloader = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"\n训练样本数：{len(wiki_dataset)}，Batch Size：{batch_size}")

# -------------------------- 5. 模型训练配置（适配官方数据集） --------------------------
hidden_size = 512
num_layers = 2
dropout = 0.3
lr = 0.001
epochs = 30  # 官方数据量大，30轮足够收敛（比之前少，避免过拟合）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextGeneratorLSTM(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------- 6. 模型训练（官方数据集训练更稳定） --------------------------
print(f"\n开始训练（设备：{device}，官方 wikitext-2 数据集）...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_input, batch_target in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        output, _ = model(batch_input)
        
        # 调整维度适配损失函数
        output = output.reshape(-1, vocab_size)
        batch_target = batch_target.reshape(-1)
        
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # 每100个batch打印一次中间损失（避免输出过多）
        if batch_count % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch {batch_count}, Current Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    print(f"Epoch [{epoch+1}/{epochs}] 结束，Average Loss: {avg_loss:.4f}")

# 保存模型（包含数据集映射，方便后续加载）
model_path = "wiki_text_generator_lstm.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "char2idx": char2idx,
    "idx2char": idx2char,
    "vocab_size": vocab_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers
}, model_path)
print(f"\n模型保存完成：{model_path}")

# -------------------------- 7. 文本生成功能（保持不变，适配官方数据集） --------------------------
def generate_text(
    model, char2idx, idx2char, start_text, generate_len=300, temperature=0.7
):
    model.eval()
    generated_text = list(start_text)
    device = next(model.parameters()).device
    
    # 预处理起始文本
    start_idx = [char2idx[c] for c in generated_text if c in char2idx]
    if not start_idx:
        start_idx = [random.randint(0, vocab_size-1)]
    
    input_tensor = torch.nn.functional.one_hot(
        torch.tensor(start_idx, dtype=torch.long),
        num_classes=vocab_size
    ).float().unsqueeze(0).to(device)
    
    # 初始化隐藏状态
    hidden = None
    _, hidden = model(input_tensor, hidden)
    
    # 逐字符生成
    current_char = generated_text[-1]
    for _ in range(generate_len):
        current_idx = char2idx.get(current_char, random.randint(0, vocab_size-1))
        input_tensor = torch.nn.functional.one_hot(
            torch.tensor([current_idx], dtype=torch.long),
            num_classes=vocab_size
        ).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
        
        # 温度调节+采样
        output = output / temperature
        probs = torch.softmax(output, dim=-1)
        next_idx = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        next_char = idx2char[next_idx]
        
        generated_text.append(next_char)
        current_char = next_char
    
    return "".join(generated_text)

# -------------------------- 8. 测试生成效果（基于官方数据集训练的模型） --------------------------
if __name__ == "__main__":
    # 测试不同起始文本（维基百科风格，如科学、历史主题）
    test_start_texts = [
        "Artificial intelligence is",
        "The history of ancient Greece",
        "Photosynthesis is the process by which",
        "In computer science, machine learning"
    ]
    
    print("\n" + "="*100)
    print("官方 wikitext-2 数据集训练模型 - 文本生成结果")
    print("="*100)
    for start_text in test_start_texts:
        generated = generate_text(
            model=model,
            char2idx=char2idx,
            idx2char=idx2char,
            start_text=start_text,
            generate_len=250,
            temperature=0.65  # 平衡通顺度和多样性
        )
        print(f"\n【起始文本】：{start_text}")
        print(f"【生成文本】：\n{generated}")
        print("-"*100)