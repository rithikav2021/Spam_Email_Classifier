# ===============================
# Spam Email Classifier
# ===============================

# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("First 5 Rows:")
print(df.head())

# -------------------------------
# Check Missing Values
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Convert Labels
# ham = 0
# spam = 1
# -------------------------------
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nAfter Label Encoding:")
print(df.head())

# -------------------------------
# Split Dataset
# -------------------------------
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data :", len(X_train))
print("Testing Data :", len(X_test))

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nTF-IDF Completed")

# -------------------------------
# Train Model
# -------------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel Saved Successfully!")