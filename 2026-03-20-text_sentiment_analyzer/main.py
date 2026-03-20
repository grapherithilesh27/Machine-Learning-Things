# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Load the dataset
train_df = pd.read_csv('train.csv')
# Preprocess the text data
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_words)
train_df['text'] = train_df['text'].apply(preprocess_text)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
# Evaluate the model
accuracy = clf.score(X_test_vectorized, y_test)
print('Model Accuracy:', accuracy)