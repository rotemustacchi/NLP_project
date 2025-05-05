# Toxic Language Detection in Online Chats 🧠💬

Detecting toxic and non-toxic messages using classical and transformer-based NLP models.

---

## 🔗 Run in Google Colab  
Click below to run the full project in your browser (no installation needed):

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZAjqivhGJRA4Go_x1mpitkPvCJxMnPHV?usp=sharing)

---

## 🗂️ Project Structure
- `main.py` — Full pipeline for training and evaluation.
- `data/train.csv` — Toxic comment dataset (from Kaggle).
- `requirements.txt` — Required libraries.
- `notebooks/` — Optional Jupyter or Colab notebooks.
- `graphical_abstract/` — Project overview image.
- `presentations/` — Slides for proposal, progress, final.

---

## 🧪 Models Used
| Model              | Description |
|-------------------|-------------|
| Naive Bayes        | Baseline using TF-IDF |
| Logistic Regression| Stronger classical ML model |
| DistilBERT         | Transformer fine-tuned on small dataset |

---

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix

---

## 📁 Dataset
Source: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

---

## ✍️ Authors
- [Your Name(s) Here]

---

## 📌 Notes
- For local execution: install requirements and place `train.csv` in the `/data/` folder.
- For best results, use a GPU when running DistilBERT.

---
