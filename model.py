# model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("enron_spam_data.csv")
print("Columns:", data.columns)

# Adjust based on your dataset's columns
# Some datasets have 'Message' and 'Spam/Ham' instead of 'text' and 'label'
data = data.rename(columns={
    'Message': 'text',
    'Spam/Ham': 'label'
})

# Drop missing values
data = data[['text', 'label']].dropna()

# Convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
pickle.dump(model, open('email_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved successfully!")
