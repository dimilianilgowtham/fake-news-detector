# 📰 Fake News Detector

A Machine Learning web app that detects whether a news article is **Real or Fake** using Natural Language Processing (TF-IDF) and Logistic Regression.

🔗 **Live Demo:**  
👉 [Try the App on Streamlit](https://dimilianilgowtham-fake-news-detector.streamlit.app)

---

## 🚀 Features

- Paste news article or headline and get prediction
- Shows whether it's likely **REAL** or **FAKE**
- Displays model's confidence level
- Clean, interactive web interface (built using Streamlit)

---

## 📁 Project Structure

📦 fake-news-detector/
├── app.py # Streamlit app
├── train_model.py # Model training script
├── Fake.csv / True.csv # Dataset files
├── vectorizer.pkl # TF-IDF vectorizer (saved)
├── fake_news_model.pkl # Trained model (saved)
├── requirements.txt # Dependencies for deployment
└── README.md # Project documentation

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## ⚙️ How It Works

1. Combine Fake and True news datasets
2. Vectorize text data using **TF-IDF**
3. Train a **Logistic Regression** model
4. Save the model and vectorizer with `joblib`
5. Build an interactive front-end using **Streamlit**
6. Deploy to the web with Streamlit Cloud

---

## ✨ Author

**Anil Gowtham**  
GitHub: [@dimilianilgowtham](https://github.com/dimilianilgowtham)
