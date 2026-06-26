import joblib

# Load saved model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

print("===== Spam Email Classifier =====")

while True:
    message = input("\nEnter your message: ")

    # Convert text
    message_vector = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        print("🚨 Spam Message")
    else:
        print("✅ Not Spam")

    choice = input("\nDo you want to test another message? (y/n): ")

    if choice.lower() != "y":
        print("\nThank you!")
        break