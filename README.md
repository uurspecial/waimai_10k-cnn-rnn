# waimai_10k-cnn-rnn
[PPT](https://docs.google.com/presentation/d/1Oz3BQTuh-iVEZS2bIb4JVtBQyrDyal-aX_NWjsgQ3mo/edit?usp=sharing)
# Character-level CNN + BiLSTM for Chinese Waimai Reviews

本專案使用 **字元級 (character-level) CNN + BiLSTM** 模型對中文外送評論做二分類（好評 / 差評）。

---

## 1. 資料集

- Kaggle：「Waimai 10k」中文外送評論資料集  
  🔗 [https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/)  
- CSV 內容：
  - `review`：中文評論文字
  - `label`：標籤（0 = 差評, 1 = 好評）

---

## 2. 模型架構

**CharCNN + BiLSTM** 流程：


- Embedding：將字元轉換為向量（128 維）
- CNN：卷積提取 n-gram 特徵（kernel_size=5, filters=128）
- MaxPool：降低序列長度，增加特徵抽象
- BiLSTM：捕捉上下文依賴（hidden=128, bidirectional=True）
- 全連接層：Dropout=0.3, Linear → ReLU → Linear → logits
- Loss：CrossEntropyLoss
- Optimizer：Adam (lr=1e-3, weight_decay=1e-5)
- 技術：gradient clipping=1.0, early stopping=patience 3

**資料處理**：

- 字元級索引表，PAD=0，UNK=最後一個 index
- 動態 padding（collate_fn）
- 訓練集 / 測試集切分：70% / 30%

---

## 3. 使用方法

```bash
# 安裝依賴
pip install torch pandas scikit-learn

# 執行訓練
python char_cnn_bilstm.py


