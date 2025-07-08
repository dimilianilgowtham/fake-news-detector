import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.subheader("Paste a news article or headline below:")

# Text input box
user_input = st.text_area("Enter the news text here", height=200)

# Prediction
if st.button("Predict"):
    if user_input.strip() != "":
        # Transform input
        input_vector = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vector)
        proba = model.predict_proba(input_vector)[0]
        confidence = max(proba) * 100

        # Show result
        if prediction[0] == 1:
            st.success("🟢 This is likely a **REAL** news article.")
        else:
            st.error("🔴 This is likely a **FAKE** news article.")

        # Show confidence
        st.info(f"🔍 Confidence: {confidence:.2f}% | [FAKE: {proba[0]*100:.2f}%, REAL: {proba[1]*100:.2f}%]")
    else:
        st.warning("⚠️ Please enter some text to predict.")
