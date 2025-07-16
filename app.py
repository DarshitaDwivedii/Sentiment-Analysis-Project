import streamlit as st
import pandas as pd
import joblib
import re
import contractions
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import io

# --- NLTK Setup ---
# This is the most reliable way to ensure NLTK data is available on Streamlit's server.
# We are directly downloading the required packages. NLTK is smart enough not to
# re-download if the data is already present.
@st.cache_resource
def download_nltk_data():
    """
    Downloads all necessary NLTK data models for the app.
    """
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Run the download function
nltk_data_available = download_nltk_data()


# --- Load the Upgraded Model ---
@st.cache_resource
def load_model():
    """
    Loads the saved model pipeline from disk.
    Cached to ensure it's only loaded once per session.
    """
    try:
        model = joblib.load('best_ml_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'best_ml_model.pkl' is in the same directory as app.py.")
        return None

# Load the entire model pipeline
best_pipeline_model = load_model()


# --- Initialize Stopwords and Lemmatizer AFTER download ---
# It's safer to initialize these after confirming NLTK data is available.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# --- Data Cleaning and Prediction Functions ---
def data_cleaning(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    negation_words = {'not', 'no', 'never', 'ain\'t'}
    filtered_tokens = [word for word in tokens if word not in stop_words or word in negation_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)


def predict_sentiment(review_text):
    if best_pipeline_model is None:
        return "Model not loaded"
    cleaned_text = data_cleaning(review_text)
    prediction = best_pipeline_model.predict([cleaned_text])
    return "Positive" if prediction[0] == 1 else "Negative"


def find_text_heavy_column(df):
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(text_cols) == 0: return None
    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_cols}
    if not avg_lengths: return None
    return max(avg_lengths, key=avg_lengths.get)


# --- Streamlit App UI ---
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Upgraded Sentiment Analysis Service 📊")
st.write("Powered by a highly-tuned Machine Learning model (LinearSVC with N-grams).")

# Only show the main app if the resources are available
if nltk_data_available and best_pipeline_model is not None:
    st.write("Analyze a single review or upload a CSV file for batch processing.")

    # Single Review Analysis
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

    # Batch File Upload and Analysis
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
                    progress_bar = st.progress(0, text="Starting analysis...")
                    predictions = []
                    for i, row in enumerate(df[review_column]):
                        predictions.append(predict_sentiment(row))
                        progress_bar.progress((i + 1) / len(df), text=f"Processing row {i+1}/{len(df)}")
                    
                    df['predicted_sentiment'] = predictions
                    
                    st.success("Analysis complete!")
                    st.dataframe(df.head())

                    @st.cache_data
                    def convert_df_to_csv(df_to_convert):
                        return df_to_convert.to_csv(index=False).encode('utf-8')

                    csv_output = convert_df_to_csv(df)

                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_output,
                        file_name=f'sentiment_predictions_{uploaded_file.name}',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.error("Application failed to start. Could not load necessary resources. Please check the deployment logs.")