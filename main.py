# Toxic Language Detection Project

# Install needed packages (uncomment if needed)
# !pip install pandas numpy scikit-learn matplotlib transformers torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load Dataset
print("Loading data...")
try:
    df = pd.read_csv('data/train.csv')
except:
    print("Please manually download 'train.csv' from Kaggle (Jigsaw Toxic Comment Challenge) and place it in the /data folder.")
    exit()

# Step 2: Data Preparation
print("Preparing data...")
df['toxic'] = df['toxic'].astype(int)
df = df[['comment_text', 'toxic']]
X = df['comment_text']
y = df['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Baseline - Naive Bayes
print("Training Naive Bayes baseline...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_preds))

# Step 4: Logistic Regression
print("Training Logistic Regression...")
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_tfidf, y_train)
logreg_preds = logreg_model.predict(X_test_tfidf)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_preds))

# Step 5: DistilBERT Fine-tuning
print("Preparing DistilBERT model...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

small_df = df.sample(5000, random_state=42)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    small_df['comment_text'].tolist(), small_df['toxic'].tolist(), test_size=0.2
)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ToxicDataset(train_encodings, train_labels)
test_dataset = ToxicDataset(test_encodings, test_labels)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)
print(f"Using device: {device}")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    #evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Training DistilBERT...")
trainer.train()

print("Evaluating DistilBERT...")
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=-1)

print("\nDistilBERT Classification Report:")
print(classification_report(test_labels, preds))

# Step 6: Confusion Matrix 
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, logreg_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Toxic', 'Toxic']).plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print("Done!")
