import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('spam_detection_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# UI layout
st.set_page_config(page_title="Email Spam Classifier", layout="centered")
st.title("📧 Email Spam Classifier")

st.markdown("Enter the content of the email, and find out if it's **Spam** or **Not Spam**!")

user_input = st.text_area("✉️ Email Text Here:", height=200)

if st.button("🧠 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some email text.")
    else:
        # Vectorize the input and predict
        input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(input_transformed)

        # Output result
        if prediction[0] == "Spam":
            st.error("❌ Spam Email")
        else:
            st.success("✅ Not Spam Email")
