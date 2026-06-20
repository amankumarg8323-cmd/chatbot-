import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Mock Intent Dataset
data = {
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning"
    ],
    "goodbye": [
        "bye",
        "see you",
        "good night"
    ],
    "internship": [
        "tell me about internship",
        "what is internship",
        "internship details"
    ],
    "skills": [
        "required skills",
        "what skills are needed",
        "skills for ai ml"
    ]
}

# Prepare training data
X = []
y = []

for intent, texts in data.items():
    for text in texts:
        X.append(text)
        y.append(intent)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_vec, y)

# Rule-Based Responses
responses = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a nice day!",
    "internship": "This AI/ML internship helps you learn ML, NLP, and Data Science.",
    "skills": "Python, Machine Learning, Data Analysis, and NLP are useful skills."
}

print("Hybrid Chatbot Started (type 'exit' to stop)")

while True:
    user_input = input("You: ").lower()

    if user_input == "exit":
        print("Bot: Goodbye!")
        break

    # Rule-Based Matching
    matched = False
    for intent, texts in data.items():
        if user_input in texts:
            print("Bot:", responses[intent])
            matched = True
            break

    # ML Prediction
    if not matched:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        print("Bot:", responses[prediction])
