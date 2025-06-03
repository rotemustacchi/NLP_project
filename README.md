# 🧪 Toxic Language Detection

Detecting toxic messages in online chat using multiple NLP models and zero-shot classification.

---

## 📂 Dataset

- Source: [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)  
- 150,000+ comments labeled as toxic (1) or non-toxic (0)

---

## 🧠 Models Used

| Model                     | Description                                  |
|--------------------------|----------------------------------------------|
| Naive Bayes              | Baseline with TF-IDF                         |
| Logistic Regression      | Classic ML model with TF-IDF features        |
| SVM (LinearSVC)          | Margin-based classifier using TF-IDF         |
| DistilBERT               | Fine-tuned transformer via Hugging Face      |
| Zero-Shot Classification | BART-based (facebook/bart-large-mnli)        |

---

## 🛠️ Technologies

- Python, pandas, numpy, matplotlib  
- scikit-learn (for classic ML models)  
- Hugging Face Transformers (DistilBERT & BART)  
- PyTorch  
- Confusion matrices, zero-shot pipeline

---

## 🧪 Evaluation Metrics

- Accuracy  
- Precision and Recall (for Toxic class)  
- F1-Score  
- Confusion Matrices

---

## 📊 Sample Results

| Model               | Precision (Toxic) | Recall (Toxic) | Accuracy |
|--------------------|-------------------|----------------|----------|
| Naive Bayes        | ~65%              | ~55%           | ~91%     |
| Logistic Regression| 72%               | 60%            | 93%      |
| SVM (LinearSVC)    | ~72%              | ~60%           | 93%      |
| DistilBERT         | 79%               | 71%            | 95%      |

---

## 🚀 Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main detection script
python toxic_detector.py
