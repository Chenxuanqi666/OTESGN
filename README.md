# OTESGN: Optimal Transport Enhanced Syntactic-Semantic Graph Networks for Aspect-Based Sentiment Analysis

This repository contains the implementation of the paper:

**OTESGN: Optimal Transport Enhanced Syntactic-Semantic Graph Networks for Aspect-Based Sentiment Analysis**

---

## 📂 Project Structure

```text
OTESGN/
│
├── bert-base-uncased/              # Pre-trained BERT-Base-Uncased model (to be downloaded)
│
├── dataset/                        # Datasets for ABSA
│   ├── Laptops_corenlp/
│   ├── Restaurants_corenlp/
│   ├── Tweets_corenlp/
│   └── preprocess_data.py
│
├── log/                            # Training logs (generated during training)
│
├── models/                         # Model-related scripts
│   ├── get_attention_score.py
│   ├── OTESGN.py
│   ├── wo_Graph_Attention.py
│   └── wo_SyntacticMask.py
│
├── state_dict/                     # Directory for saving model checkpoints
│
├── data_utils.py
├── prepare_vocab.py
└── train.py                        # Entry point for training
```


---

## 🔧 Requirements

The following Python packages are required:
```text
json
os
sys
copy
random
logging
argparse
torch
numpy
sklearn
time
transformers
```

You can install them with:

pip install torch numpy scikit-learn transformers

---

## 📥 Pre-trained Model

Please download **BERT-Base-Uncased** from [Hugging Face](https://huggingface.co/bert-base-uncased) or the [official BERT repository](https://github.com/google-research/bert).  
Place the model files under the directory:

./bert-base-uncased/


---

## 🚀 How to Run

Simply execute:

```bash
python train.py
```
---

## 📖 Citation
```text
 If you find our work helpful, please cite the following paper:
    @inproceedings{Liao2025OTESGN,
      author = {Xinfeng Liao and Xuanqi Chen and Lianxi Wang and Jiahuan Yang and Zhuowei Chen and Ziying Rong},
      title = {OTESGN: Optimal Transport Enhanced Syntactic-Semantic Graph Networks for Aspect-Based Sentiment Analysis},
      booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
      year = {2025},
      address = {Washington, DC, USA},
      publisher = {IEEE},
      pages = {to appear}
}
```