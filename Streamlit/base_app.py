"""
Simple Streamlit webserver application for serving developed classification models.

Author: ExploreAI Academy.

Note:
---------------------------------------------------------------------
Please follow the instructions provided within the README.md file
located within this directory for guidance on how to use this script
correctly.
---------------------------------------------------------------------

Description: This file is used to launch a minimal streamlit web
application. You are expected to extend the functionality of this script
as part of your predict project.

For further help with the Streamlit framework, see:
https://docs.streamlit.io/en/latest/
"""

# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import nltk

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

# Load the pickled model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('pickled_files/model_and_vectorizer.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['vectorizer']
    except FileNotFoundError:
        st.error("Model file not found! Please run the notebook first to train and save the model.")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# Initialize text processing components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_text_pro(text: str) -> str:
    """
    Preprocess text for classification - same function as used in training
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    # 3. Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 4. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Tokenize
    tokens = word_tokenize(text)
    # 6. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # 7. Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def main():
    """News Classifier App with Streamlit"""
    
    # Creates a main title and subheader on your page
    st.title("News Classifier Project")
    st.subheader("Analyzing News Articles")
    
    # Check if model is loaded
    if model is None or vectorizer is None:
        st.error("Model not loaded. Please run the model.ipynb notebook first to train and save the model.")
        return
    
    # Creating sidebar with selection box
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("""
        ### About This App
        This news classifier can categorize news articles into one of five categories:
        - **Business** - Financial news, company updates, market analysis
        - **Technology** - Tech innovations, software, hardware, digital trends
        - **Sports** - Games, athletes, tournaments, sports news
        - **Education** - Schools, learning, academic research, educational policy
        - **Entertainment** - Movies, music, celebrities, TV shows, cultural events
        
        ### How It Works
        1. Enter your news article text in the text area
        2. Click "Classify" to predict the category
        3. The model uses machine learning to analyze the text and predict the most likely category
        
        ### Model Information
        - **Algorithm**: Logistic Regression
        - **Text Processing**: TF-IDF Vectorization with preprocessing
        - **Training Data**: News articles from various categories
        """)
    
    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")

        # --- Function to clear text ---
        def clear_text():
            st.session_state.article_input = ""

        # --- User input text area ---
        news_text = st.text_area(
            "Enter News Article Text",
            placeholder="Paste your news article here...",
            height=200,
            key="article_input"
        )

        # --- Two buttons side-by-side ---
        col1, col2 = st.columns([1, 1])
        with col1:
            classify_button = st.button("🔍 Classify")
        with col2:
            clear_button = st.button("🧹 Clear Text", on_click=clear_text)

        # --- Classification logic ---
        if classify_button:
            if news_text.strip():
                try:
                    processed_text = process_text_pro(news_text)
                    vect_text = vectorizer.transform([processed_text])
                    prediction = model.predict(vect_text)[0]
                    probabilities = model.predict_proba(vect_text)[0]
                    confidence = max(probabilities) * 100

                    st.success(f"**Predicted Category:** {prediction.title()}")
                    st.info(f"**Confidence:** {confidence:.1f}%")

                    st.subheader("Category Probabilities:")
                    prob_df = pd.DataFrame({
                        'Category': model.classes_,
                        'Probability': probabilities * 100
                    }).sort_values('Probability', ascending=False)
                    st.bar_chart(prob_df.set_index('Category'))

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
            else:
                st.warning("Please enter some text to classify.")


            # --- Clear Text Button Logic ---
            if clear_button:
                # Reset input and rerun app
                st.session_state["article_input"] = ""
                st.experimental_rerun()


# Required to let Streamlit instantiate our web app
if __name__ == '__main__':
    main()