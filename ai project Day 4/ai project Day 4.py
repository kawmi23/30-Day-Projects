import sys
import subprocess

# This line automatically installs numpy, pandas, and scikit-learn straight into IDLE
required_libraries = ["numpy", "pandas", "scikit-learn"]
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"Installing {lib} for you now... Please wait a moment.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"{lib} installed successfully!")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (using a spam email dataset)
# Here we'll use a common dataset available on Kaggle or UCI, stored locally as 'spam.csv'
#df = pd.read_csv('spam.csv', encoding='latin-1')
# PASTE THIS RIGHT BELOW WHERE YOU REMOVED THE PREVIOUS LINE:
import io

mock_data = """v1,v2
ham,Hey are we still meeting up for coffee later?
spam,CONGRATULATIONS You won a thousand dollar gift card Click here now
ham,Dont forget your database systems assignment is due tomorrow
spam,Urgent security notice Your bank account has been locked Verify here
ham,Hi mom I will be home late tonight for dinner
spam,Get cheap loans instantly with zero percent interest click link"""

# This reads the text data directly into Pandas without needing a file!
df = pd.read_csv(io.StringIO(mock_data))

# Preprocess the dataset
df = df[['v1', 'v2']]  # We are only interested in the 'v1' (label) and 'v2' (message) columns
df.columns = ['label', 'message']  # Rename the columns for clarity
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Map 'ham' (non-spam) to 0 and 'spam' to 1

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert the text data to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform the training data
X_test_vec = vectorizer.transform(X_test)  # Only transform the test data

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
