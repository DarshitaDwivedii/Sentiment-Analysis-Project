# Sentiment Analysis of IMDB Movie Reviews

This project provides a complete end-to-end workflow for sentiment analysis on the IMDB movie review dataset. It includes data cleaning, exploratory data analysis, training and tuning a high-performance machine learning model, and deploying the model as an interactive web application using Streamlit.

![Streamlit App Demo](https://sentiment-analysis-project-p3.streamlit.app/)
*(Note: You will need to create this GIF yourself and upload it to your repo for it to display)*


## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)


## ✨ Features

- **Data Preprocessing:** Robust text cleaning pipeline to handle HTML tags, URLs, contractions, and stopwords.
- **Advanced Model Training:** Utilizes `scikit-learn`'s `Pipeline` and `GridSearchCV` to find the optimal hyperparameters for a `LinearSVC` model with TF-IDF vectorization, including n-grams.
- **High-Performance Model:** The final tuned model achieves **~91% accuracy** on the test set.
- **Interactive Web Application:** A user-friendly app built with Streamlit that allows for:
    - Real-time sentiment analysis of single text inputs.
    - Batch processing by uploading a CSV file.
    - Automatic detection of the review column in uploaded datasets.
- **Reproducible Workflow:** The entire process from data loading to model creation is documented in the `Sentiment_Analysis_Project.ipynb` notebook.


## 📂 Project Structure

The repository is organized as follows:
SENTIMENT-ANALYSIS-PROJECT/
│
├── .venv/ # Virtual environment files
├── .git/ # Git repository data
│
├── app.py # The Streamlit web application script
├── best_ml_model.pkl # Saved file for the trained model pipeline
├── IMDB Dataset.csv # The dataset used for training and testing
├── requirements.txt # Required Python libraries for the project
├── Sentiment_Analysis_Project.ipynb # Jupyter notebook with the full analysis
└── README.md # This documentation file


## 🛠️ Technical Stack

- **Language:** Python 3.10
- **Core Libraries:**
    - **Data Manipulation:** Pandas, NumPy
    - **Machine Learning:** Scikit-learn
    - **NLP:** NLTK, Contractions
    - **Web Framework:** Streamlit
    - **Visualization:** Matplotlib, Seaborn, WordCloud
- **Development Environment:** Jupyter Notebook, Visual Studio Code


## 🚀 Installation and Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DarshitaDwivedii/Sentiment-Analysis-Project.git
    cd SENTIMENT-ANALYSIS-PROJECT
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\Activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

There are two ways to explore this project:

### 1. Running the Web Application

To launch the interactive Streamlit app, run the following command in your terminal:

```bash
streamlit run app.py
```
Your web browser will automatically open to the application's URL. You can then enter a review or upload a CSV file to see the model in action.

### 2. Exploring the Jupyter Notebook
To see the detailed steps of data analysis, model training, and evaluation, you can run the Jupyter notebook:
1. Make sure your virtual environment is active.
2. Launch Jupyter Lab (or Notebook)
3. Open the Sentiment_Analysis_Project.ipynb file and run the cells sequentially.

## 📈 Model Performance
The final machine learning model was selected after an exhaustive search using GridSearchCV. The best performing model is a Linear Support Vector Classifier (LinearSVC) using TF-IDF features, including both unigrams and bigrams.

Final Test Set Results:
- Metric	Score
- Accuracy	0.91
- Precision	0.91
- Recall	0.91
- F1-Score	0.91
The model demonstrates strong, balanced performance in classifying both positive and negative reviews.

## 🔮 Future Improvements
The potential future enhancements could include:
- **Advanced Deep Learning**: Implementing a state-of-the-art Transformer-based model (like BERT or DistilBERT) to potentially increase accuracy to the 94-95% range.
- **Aspect-Based Sentiment Analysis**: Extending the model to identify not just the overall sentiment, but the sentiment towards specific aspects (e.g., "The acting was great, but the plot was terrible").
- **CI/CD Pipeline**: Setting up a continuous integration/continuous deployment pipeline to automatically test and deploy changes to the Streamlit application.