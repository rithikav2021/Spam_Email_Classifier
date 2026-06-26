# Spam Email Classifier

## 📌 Project Overview

This project is a Machine Learning based Spam Email Classifier developed using Python and Scikit-learn. It classifies SMS messages as **Spam** or **Not Spam (Ham)** using the Naive Bayes algorithm and TF-IDF Vectorization.

## 🚀 Features

* Detects Spam and Non-Spam messages
* Uses TF-IDF for text vectorization
* Trained using Naive Bayes Classifier
* Command-line interface for prediction
* Saves trained model using Joblib

## 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib

## 📂 Project Structure

Spam_Email_Classifier/

├── dataset/
│ └── spam.csv

├── model/
│ ├── spam_model.pkl
│ └── vectorizer.pkl

├── train_model.py

├── app.py

├── requirements.txt

└── README.md

## ▶️ How to Run

1. Install required libraries:

pip install -r requirements.txt

2. Train the model:

py train_model.py

3. Run the application:

py app.py

## 📊 Model Performance

* Algorithm: Multinomial Naive Bayes
* Accuracy: Approximately **97%**

## 📷 Sample Output

Spam Message Example:

Congratulations! You won ₹10,000. Click here to claim.

Output:

Spam Message

Normal Message Example:

Hi, shall we meet tomorrow at 5 PM?

Output:

Not Spam

## 👨‍💻 Author

**Rithika**

AI Internship Project

Developed using Python and Machine Learning.

