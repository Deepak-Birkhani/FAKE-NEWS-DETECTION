import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("news_vectorizer.pkl")

def predict_news(text: str) -> str:
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)
    return "✅ Real News" if prediction == 1 else "❌ Fake News"

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news article below and I will tell you whether it's Real or Fake.")

user_input = st.text_area("Paste your news content here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        result = predict_news(user_input)
        st.success(f"Prediction: {result}")
