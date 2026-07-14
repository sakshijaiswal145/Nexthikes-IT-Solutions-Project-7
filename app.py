import streamlit as st
import pickle
import re

# Page configuration
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)

# Load model and vectorizer
model = pickle.load(open("disaster_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Title
st.title("🚨 Disaster Tweet Classification")
st.markdown(
    "Enter a tweet below and the model will predict whether it is a **Real Disaster Tweet** or **Non-Disaster Tweet**."
)

# Input box
tweet = st.text_area(
    "Enter Tweet",
    placeholder="Example: Massive earthquake hits California..."
)

# Prediction
if st.button("Predict"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned_tweet = clean_text(tweet)

        tweet_vector = tfidf.transform([cleaned_tweet])

        prediction = model.predict(tweet_vector)[0]

        if prediction == 1:
            st.error("🚨 This is a REAL DISASTER Tweet.")
        else:
            st.success("✅ This is a NON-DISASTER Tweet.")

# Sidebar
st.sidebar.title("Project Information")
st.sidebar.write("""
**Project:** Disaster Tweet Classification

**Algorithm Used:** Logistic Regression

**Dataset Size:** 10,873 Tweets

**Tech Stack:**
- Python
- NLP
- TF-IDF
- Scikit-Learn
- Streamlit
""")