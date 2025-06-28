# fake_news_app.py
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved pipeline
@st.cache_resource
def load_model():
    return joblib.load('fake_news_detector_pipeline.pkl')

# Your text cleaning function
def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Set up the app
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("🔍 Fake News Detector")
st.markdown("Upload text or paste a news article to check its authenticity")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    - Model: **Logistic Regression** (99% accuracy)
    - Features: TF-IDF text analysis
    - Training data: 5,000+ labeled articles
    """)

# Input options
tab1, tab2 = st.tabs(["Text Input", "File Upload"])

with tab1:
    user_input = st.text_area("Paste news article here:", height=200)
    if st.button("Analyze Text"):
        if user_input:
            pipeline = load_model()
            prediction = pipeline.predict([user_input])[0]
            proba = pipeline.predict_proba([user_input]).max()
            
            if prediction == "fake":
                st.error(f"🚨 Fake News Detected ({proba:.0%} confidence)")
            else:
                st.success(f"✅ Real News ({proba:.0%} confidence)")
        else:
            st.warning("Please enter some text")

with tab2:
    uploaded_file = st.file_uploader("Upload TXT or CSV", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
        else:
            df = pd.read_csv(uploaded_file)
            text = df.iloc[0, 0]  # Assuming text is in first column
        
        pipeline = load_model()
        prediction = pipeline.predict([text])[0]
        
        st.subheader("Analysis Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Confidence", f"{pipeline.predict_proba([text]).max():.0%}")

# Adding some sample text
expander = st.expander("💡 Try these examples")
expander.write("""
**Fake News Examples:**
- "The government is hiding alien technology in Area 51"
- "Vaccines contain microchips to track citizens"

**Real News Examples:**
- "Scientists confirm climate change is accelerating"
- "New education bill passed by parliament"
""")

# Footer
st.markdown("---")
st.caption("Model last updated: 2023-08-15 | [Download Model](fake_news_detector_pipeline.pkl)")