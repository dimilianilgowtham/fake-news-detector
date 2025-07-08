import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Balance the dataset
min_len = min(len(fake), len(real))
fake = fake.sample(min_len, random_state=42)
real = real.sample(min_len, random_state=42)

# Combine and shuffle
data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data
X = data["text"]
y = data["label"]

# FIT the vectorizer properly
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save the fitted vectorizer and model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "fake_news_model.pkl")

# Balance the dataset
min_len = min(len(fake), len(real))
fake = fake.sample(min_len, random_state=42)
real = real.sample(min_len, random_state=42)

print("✅ Model and vectorizer saved successfully and properly fitted!")
print(data['label'].value_counts())
from sklearn.metrics import classification_report

y_pred = model.predict(X_vectorized)
print(classification_report(y, y_pred))


