# char_cnn_bilstm.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import f1_score, accuracy_score

# ======================
# 0. 讀取數據
# ======================
# 參考來源：
# - Kaggle中文外送評論示例資料
df = pd.read_csv('/home/jovyan/waimai_10k.csv')
labels = list(df['label'].values)
txt_list = list(df['review'].values)

# ======================
# 1. 建立字元表
# ======================
# 參考來源：
# - Zhang et al., 2015, Character-level CNN for text classification
# - Chinese-Text-Classification-Pytorch GitHub: https://github.com/649453932/Chinese-Text-Classification-Pytorch
char_set = set()
for txt in txt_list:
    for char in txt:
        char_set.add(char)

char_list = list(char_set)
n_chars = len(char_list) + 2  # +1 for <UNK>, +1 for <PAD>
PAD_ID = 0
UNK_ID = len(char_list) + 1

print("詞表大小：", n_chars)

# 字元 -> 索引
char2idx = {c: i+1 for i, c in enumerate(char_list)}  # 0留給PAD
char2idx["<UNK>"] = UNK_ID

def text_to_tensor(text):
    # 將字串轉成索引 tensor
    ids = []
    for ch in text:
        ids.append(char2idx.get(ch, UNK_ID))
    return torch.tensor(ids, dtype=torch.long)

# ======================
# 2. Dataset / DataLoader
# ======================
# 參考來源：
# - PyTorch 官方 tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# - 動態 padding collate_fn 參考 Chinese-Text-Classification-Pytorch repo
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        txt, label = self.data[idx]
        return txt, label

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(x) for x in texts]
    max_len = max(lengths)
    padded = torch.full((len(texts), max_len), PAD_ID, dtype=torch.long)
    for i, x in enumerate(texts):
        padded[i, :len(x)] = x
    return padded, torch.tensor(labels, dtype=torch.long), lengths

# 準備資料
all_data = [(text_to_tensor(txt), torch.tensor(label, dtype=torch.long)) for txt, label in zip(txt_list, labels)]
random.shuffle(all_data)
split = int(0.7 * len(all_data))
train_data, test_data = all_data[:split], all_data[split:]

train_dataset = ReviewDataset(train_data)
test_dataset = ReviewDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ======================
# 3. 模型設計
# ======================
# 架構參考：
# - CNN + MaxPool 特徵抽取: Zhang et al., 2015
# - BiLSTM 捕捉序列依賴: PyTorch tutorial + Chinese-Text-Classification-Pytorch
class CharCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_filters=128, kernel_size=5, lstm_hidden=128, num_classes=2, dropout=0.3):
        super().__init__()
        # Embedding 層: 將字元索引轉成向量
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        # CNN 層: 1D 卷積提取字元 n-gram 特徵
        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        # MaxPool 降低序列長度，增加特徵抽象
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # BiLSTM 捕捉上下文依賴
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # 全連接層分類
        self.fc1 = nn.Linear(2*lstm_hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x, lengths=None):
        emb = self.embedding(x)              # (B, T, emb_dim)
        emb = emb.permute(0, 2, 1)           # (B, emb_dim, T)
        conv_out = torch.relu(self.conv(emb))# (B, num_filters, T)
        pooled = self.pool(conv_out)         # (B, num_filters, T/2)
        pooled = pooled.permute(0, 2, 1)     # (B, T/2, num_filters)

        # LSTM
        packed_out, (h, c) = self.lstm(pooled)
        # 取最後 hidden state 作為序列表示
        out = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2*lstm_hidden)

        out = torch.relu(self.fc1(self.dropout(out)))
        logits = self.fc2(out)
        return logits

# ======================
# 4. 訓練流程
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharCNNBiLSTM(vocab_size=n_chars).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def evaluate(loader):
    model.eval()
    preds, targets = [], []
    losses = []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            losses.append(loss.item())
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(y.cpu().numpy())
    f1 = f1_score(targets, preds, average="macro")
    acc = accuracy_score(targets, preds)
    return sum(losses)/len(losses), f1, acc

best_f1 = 0
patience, wait = 3, 0

for epoch in range(20):
    model.train()
    train_losses, train_preds, train_targets = [], [], []
    for x, y, lengths in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        train_preds.extend(logits.argmax(dim=1).cpu().numpy())
        train_targets.extend(y.cpu().numpy())
    
    train_f1 = f1_score(train_targets, train_preds, average="macro")
    train_loss = sum(train_losses)/len(train_losses)
    val_loss, val_f1, val_acc = evaluate(test_loader)
    
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        wait = 0
        torch.save(model.state_dict(), "best_char_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val_f1={best_f1:.4f}.")
            break

# ======================
# 5. 推論
# ======================
model.load_state_dict(torch.load("best_char_model.pt"))
model.eval()
test_samples = ["這家店外送很快，下次還會再點", "食物太難吃了，送餐還遲到"]
for s in test_samples:
    x = text_to_tensor(s).unsqueeze(0).to(device)
    logits = model(x)
    pred = logits.argmax(dim=1).item()
    print(s, "=>", "好評" if pred==1 else "差評")
