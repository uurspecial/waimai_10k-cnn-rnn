# waimai_10k-cnn-rnn
[PPT](https://docs.google.com/presentation/d/1Oz3BQTuh-iVEZS2bIb4JVtBQyrDyal-aX_NWjsgQ3mo/edit?usp=sharing)
# Character-level CNN + BiLSTM for Chinese Waimai Reviews

本專案使用 **字元級 (character-level) CNN + BiLSTM** 模型對中文外送評論做二分類（好評 / 差評）。

---

## 1. 資料集

- Kaggle：「Waimai 10k」中文外送評論資料集  
  [https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/haoshaoyang/waimai-10k)  
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
pip install -r requirements.txt  

# 執行訓練
python char_cnn_bilstm.py
```
## 4.參考

1. **Zhang et al., 2015, Character-level CNN for Text Classification**  
   - 描述：原始字元級 CNN 模型，用於文本分類的基線架構。  
   - GitHub：[https://github.com/zhangxiangxiao/Crepe](https://github.com/zhangxiangxiao/Crepe)

2. **Chinese-Text-Classification-Pytorch**  
   - 描述：中文文本分類多模型實作範例，包括 TextCNN、RNN、BiLSTM、RCNN 等，提供完整資料前處理及模型訓練流程。  
   - GitHub：[https://github.com/649453932/Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

3. **PyTorch 官方 RNN / LSTM 教學**  
   - 描述：PyTorch 官方教學，示範如何使用 RNN、LSTM、GRU 處理序列資料，包含 pack_padded_sequence 與動態序列長度處理。  
   - 網址：[https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)



