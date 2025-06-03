# 🧪 Toxic Language Detection

Detecting toxic messages in online chat using machine learning and transformers.  
This project compares classic ML models and transformer-based models on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset.

---

## 📂 Dataset
- Source: Kaggle — Jigsaw Toxic Comment Classification Challenge  
- 150,000+ comments labeled as `toxic` (1) or `non-toxic` (0)

---

## 🧠 Models Used

| Model               | Type             | Notes                        |
|--------------------|------------------|-------------------------------|
| Naive Bayes        | Traditional ML   | TF-IDF baseline               |
| Logistic Regression| Traditional ML   | Improved precision & recall  |
| DistilBERT         | Transformer (Hugging Face) | Fine-tuned on GPU       |
| Azure Content Moderator | API | REST-based toxicity detection |

---

## 🛠️ Technologies

- `scikit-learn`, `pandas`, `numpy`, `matplotlib`  
- `transformers`, `torch`, `datasets`  
- Azure ML, Azure OpenAI, Azure Content Moderator API

---

## 🧪 Evaluation Metrics

- Accuracy  
- Precision & Recall (especially for toxic class)  
- F1-Score  
- Confusion Matrix (Logistic Regression example shown)

---

## 📊 Sample Results

| Model               | Precision (Toxic) | Recall (Toxic) | Accuracy |
|--------------------|-------------------|----------------|----------|
| Naive Bayes        | 65%               | 55%            | 91%      |
| Logistic Regression| 72%               | 60%            | 93%      |
| DistilBERT         | 79%               | 71%            | 95%      |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

