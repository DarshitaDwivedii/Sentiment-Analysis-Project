import streamlit as st
import pandas as pd
import joblib
import re
import contractions
import nltk
import io

# ==============================================================================
# SCRIPT SETUP: LOAD ALL RESOURCES AT THE VERY BEGINNING
# ==============================================================================

# --- 1. NLTK DATA DOWNLOAD ---
# We are directly calling the download commands at the top level of the script.
# This ensures they are executed once when the app starts, before any other code runs.
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")
    # Stop the app if NLTK data fails to download
    st.stop()

# --- 2. LOAD THE MACHINE LEARNING MODEL ---
# This loads the entire pipeline (vectorizer + classifier).
try:
    model_pipeline = joblib.load('best_ml_model.pkl')
except FileNotFoundError:
    st.error("Model file 'best_ml_model.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()


# --- 3. IMPORT AND INITIALIZE NLTK COMPONENTS ---
# These MUST be initialized after the download is complete.
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ==============================================================================
# APP FUNCTIONS
# ==============================================================================

def data_cleaning(text):
    """
    Cleans the input text using the same process as the training notebook.
    """
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text) # This is the line that was causing the error
    negation_words = {'not', 'no', 'never', 'ain\'t'}
    filtered_tokens = [word for word in tokens if word not in stop_words or word in negation_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)


def predict_sentiment(review_text):
    """
    Takes raw text, cleans it, and uses the loaded model pipeline to predict sentiment.
    """
    cleaned_text = data_cleaning(review_text)
    # The pipeline handles both vectorization and prediction
    prediction = model_pipeline.predict([cleaned_text])
    return "Positive" if prediction[0] == 1 else "Negative"


def find_text_heavy_column(df):
    """
    Finds the column in a DataFrame that most likely contains the text reviews.
    """
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(text_cols) == 0: return None
    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_cols}
    if not avg_lengths: return None
    return max(avg_lengths, key=avg_lengths.get)


# ==============================================================================
# STREAMLIT USER INTERFACE
# ==============================================================================

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Upgraded Sentiment Analysis Service 📊")
st.write("Powered by a highly-tuned Machine Learning model (LinearSVC with N-grams).")
st.write("Analyze a single review or upload a CSV file for batch processing.")

# --- Single Review Analysis ---
st.header("Analyze a Single Review")
default_text = "The movie was not good at all. I don't know why people are liking it so much."
user_input = st.text_area("Enter a movie review here:", default_text, height=100)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing..."):
            sentiment = predict_sentiment(user_input)
            if sentiment == "Positive":
                st.success(f"Sentiment: **{sentiment}**")
            else:
                st.error(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review to analyze.")

# --- Batch File Upload and Analysis ---
st.header("Analyze a CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        review_column = find_text_heavy_column(df)

        if review_column is None:
            st.error("Error: Could not find a suitable text column in the uploaded CSV.")
        else:
            st.info(f"Automatically detected **'{review_column}'** as the review column.")

            with st.spinner(f"Processing your file... Analyzing column '{review_column}'."):
                # Using a list comprehension for faster processing on the server
                predictions = [predict_sentiment(row) for row in df[review_column]]
                df['predicted_sentiment'] = predictions

                st.success("Analysis complete!")
                st.dataframe(df.head())

                # Prepare the file for download
                csv_output = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_output,
                    file_name=f'sentiment_predictions_{uploaded_file.name}',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"An error occurred: {e}")