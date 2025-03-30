import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
import os
import traceback

# Set page configuration
st.set_page_config(
    page_title="Hotel Review Sentiment Analyzer",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources if not already downloaded
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load model
@st.cache_resource
def load_model_files():
    # Path to saved model files
    model_path = "hotel_sentiment_model.h5"
    label_encoder_path = "label_encoder.pkl"
    
    try:
        # Check if model already exists
        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            st.success("Found model files")
            model = load_model(model_path)
            label_encoder = joblib.load(label_encoder_path)
            return model, label_encoder
        else:
            st.warning(f"Model files not found. Looking for: {model_path} and {label_encoder_path}")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

# Define text processing functions
def get_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def extract_words_with_pos(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()                     
    word_tokens = word_tokenize(text)       
    pos_tags = pos_tag(word_tokens)         
    
    words_with_pos = [(word, tag) for word, tag in pos_tags] 
    
    # Filter out stopwords and lemmatize
    lemmatized_text = [
        lemmatizer.lemmatize(word, get_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words
    ]
    
    return words_with_pos, ' '.join(lemmatized_text) if lemmatized_text else ""

# Function to predict sentiment with thresholds
def predict_sentiment_with_threshold(review_text, model, label_encoder, voc_size=10000, sent_length=20, optimal_thresholds=None):
    try:
        if optimal_thresholds is None:
            # Default thresholds if not provided
            optimal_thresholds = [0.2000, 0.1287, 0.7927]  # Consistent thresholds
        
        st.write(f"Debug: Using thresholds {optimal_thresholds}")
        
        # Process the review
        _, processed = extract_words_with_pos(review_text)
        st.write(f"Debug: Processed text: {processed[:50]}...")
        
        # Convert to one-hot encoding
        one_hot_review = [one_hot(processed, voc_size)]
        st.write(f"Debug: One-hot encoding created with length {len(one_hot_review[0])}")
        
        # Pad sequences
        padded_review = pad_sequences(one_hot_review, padding='pre', maxlen=sent_length)
        st.write(f"Debug: Padded sequence shape {padded_review.shape}")
        
        # Get raw probabilities
        st.write("Debug: About to predict...")
        predictions_prob = model.predict(padded_review)
        st.write(f"Debug: Raw prediction probabilities: {predictions_prob}")
        
        # Apply optimal thresholds
        normalized_probs = np.zeros_like(predictions_prob)
        n_classes = len(optimal_thresholds)
        
        for i in range(n_classes):
            normalized_probs[0, i] = predictions_prob[0, i] / optimal_thresholds[i]
        
        st.write(f"Debug: Normalized probabilities: {normalized_probs}")
        
        prediction_adjusted = np.argmax(normalized_probs, axis=1)
        st.write(f"Debug: Adjusted prediction index: {prediction_adjusted}")
        
        # Convert back to label
        predicted_label = label_encoder.inverse_transform(prediction_adjusted)[0]
        st.write(f"Debug: Predicted label: {predicted_label}")
        
        return predicted_label, predictions_prob[0]
    
    except Exception as e:
        st.error(f"Error in prediction function: {str(e)}")
        st.error(traceback.format_exc())
        return "Error", np.array([0, 0, 0])

# Visualize probabilities
def plot_probability_chart(probabilities, labels, thresholds):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#d9534f', '#f0ad4e', '#5cb85c']  # Red for negative, yellow for neutral, green for positive
        
        # Create two subplots - one for raw probabilities, one for threshold-adjusted scores
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Plot raw probabilities
        raw_bars = ax1.bar(labels, probabilities, color=colors)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Raw Probability')
        ax1.set_title('Raw Sentiment Probabilities')
        
        # Add probability text on bars
        for bar in raw_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Plot threshold-adjusted scores
        normalized_scores = [prob/threshold for prob, threshold in zip(probabilities, thresholds)]
        adj_bars = ax2.bar(labels, normalized_scores, color=colors)
        ax2.set_ylim(0, max(normalized_scores) * 1.1)  # Adjusted ylim for better visibility
        ax2.set_ylabel('Threshold-Adjusted Score')
        ax2.set_title('Threshold-Adjusted Scores')
        
        # Add normalized score text on bars
        for bar in adj_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.error(traceback.format_exc())
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Error creating chart", ha='center', va='center')
        return fig

# Add threshold metrics display
def display_threshold_metrics():
    col1, col2, col3 = st.columns(3)
    
    metrics = {
        "Negative": {
            "threshold": 0.2000,
            "sensitivity": 0.7341,
            "specificity": 0.7919,
            "false_positive": 0.2081
        },
        "Neutral": {
            "threshold": 0.1287,
            "sensitivity": 0.7268,
            "specificity": 0.6040,
            "false_positive": 0.3960
        },
        "Positive": {
            "threshold": 0.7927,
            "sensitivity": 0.6594,
            "specificity": 0.8304,
            "false_positive": 0.1696
        }
    }
    
    with col1:
        st.markdown("### Negative Class")
        st.metric("Threshold", f"{metrics['Negative']['threshold']:.4f}")
        st.metric("Sensitivity", f"{metrics['Negative']['sensitivity']:.4f}")
        st.metric("Specificity", f"{metrics['Negative']['specificity']:.4f}")
    
    with col2:
        st.markdown("### Neutral Class")
        st.metric("Threshold", f"{metrics['Neutral']['threshold']:.4f}")
        st.metric("Sensitivity", f"{metrics['Neutral']['sensitivity']:.4f}")
        st.metric("Specificity", f"{metrics['Neutral']['specificity']:.4f}")
    
    with col3:
        st.markdown("### Positive Class")
        st.metric("Threshold", f"{metrics['Positive']['threshold']:.4f}")
        st.metric("Sensitivity", f"{metrics['Positive']['sensitivity']:.4f}")
        st.metric("Specificity", f"{metrics['Positive']['specificity']:.4f}")

# Function to handle model file upload
def handle_model_upload():
    st.header("Upload Model Files")
    st.write("Model files not found. Please upload the required files to proceed.")
    
    model_file = st.file_uploader("Upload the model file (hotel_sentiment_model.h5)", type=["h5"])
    encoder_file = st.file_uploader("Upload the label encoder file (label_encoder.pkl)", type=["pkl"])
    
    if model_file and encoder_file:
        try:
            # Save uploaded files
            with open("hotel_sentiment_model.h5", "wb") as f:
                f.write(model_file.getbuffer())
            with open("label_encoder.pkl", "wb") as f:
                f.write(encoder_file.getbuffer())
            
            st.success("Files uploaded successfully! Please refresh the page to load the model.")
            return True
        except Exception as e:
            st.error(f"Error saving uploaded files: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    return False

# Validate review input
def validate_review(review_text):
    if not review_text or len(review_text.strip()) < 10:
        st.warning("Please enter a more detailed review (at least 10 characters) for accurate analysis.")
        return False
    return True

# Streamlit UI
def main():
    # Test TensorFlow functionality
    try:
        # Simple TensorFlow operation to check if it's working
        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        st.sidebar.success("TensorFlow is working correctly")
    except Exception as e:
        st.sidebar.error(f"TensorFlow error: {str(e)}")
    
    # Set optimal thresholds
    optimal_thresholds = [0.2000, 0.1287, 0.7927]  # [Negative, Neutral, Positive]
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2437/2437944.png", width=100)
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app uses a bidirectional LSTM neural network to analyze 
        hotel reviews and predict sentiment. The model categorizes 
        reviews as Negative, Neutral, or Positive using optimized 
        thresholds for each class.
        """
    )
    
    st.sidebar.title("How it works")
    st.sidebar.markdown(
        """
        1. Enter your hotel review in the text area
        2. Click "Analyze Sentiment"
        3. View the predicted sentiment and confidence scores
        
        The model processes text using:
        - Text cleaning
        - Stop word removal
        - Lemmatization
        - Neural network analysis
        
        Optimal thresholds are applied to each class to maximize 
        sensitivity and specificity.
        """
    )
    
    st.sidebar.title("Model Metrics")
    st.sidebar.markdown(
        """
        The model was trained on TripAdvisor hotel reviews and optimized using 
        ROC curve analysis to find the optimal threshold for each sentiment class:
        
        - **Negative**: 0.2000
        - **Neutral**: 0.1287
        - **Positive**: 0.7927
        
        These thresholds maximize the balance between sensitivity (true positive rate)
        and specificity (true negative rate).
        """
    )
    
    # Main content
    st.title("🏨 Hotel Review Sentiment Analyzer")
    st.markdown("""
    Enter a hotel review to analyze its sentiment. The model will classify the review as Positive, Neutral, or Negative
    using class-specific optimal thresholds derived from ROC curve analysis.
    """)
    
    # Create a debug expander
    debug_expander = st.expander("Debug Information (Expand for troubleshooting)", expanded=False)
    
    # Load model
    with st.spinner("Loading model..."):
        model, label_encoder = load_model_files()
    
    with debug_expander:
        st.write("Debug: Model loading check")
        if model is not None and label_encoder is not None:
            st.write("Model and encoder loaded successfully!")
            st.write(f"Label encoder classes: {label_encoder.classes_}")
            st.write(f"Model summary: {model.summary()}")
        else:
            st.error("Failed to load model or encoder")
    
    # Handle missing model files
    if model is None or label_encoder is None:
        if not handle_model_upload():
            return
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_input = st.text_area(
            "Enter your hotel review:",
            height=150,
            placeholder="Type or paste your hotel review here... (minimum 10 characters)"
        )
        
        # Analyze button
        analyze_button = st.button("Analyze Sentiment", type="primary")
    
    # Results section
    if analyze_button:
        with debug_expander:
            st.write(f"Debug: Button clicked, review input length: {len(review_input) if review_input else 0}")
        
        # Validate input
        if not validate_review(review_input):
            return
            
        with st.spinner("Analyzing sentiment..."):
            with debug_expander:
                st.write("Debug: Starting prediction process")
            
            # Predict sentiment
            try:
                sentiment, probabilities = predict_sentiment_with_threshold(
                    review_input, model, label_encoder, 
                    optimal_thresholds=optimal_thresholds
                )
                
                with debug_expander:
                    st.write(f"Debug: Prediction completed - Sentiment: {sentiment}")
                    st.write(f"Debug: Probabilities: {probabilities}")
                
                # Display results
                sentiment_colors = {
                    "Positive": "green",
                    "Neutral": "orange",
                    "Negative": "red"
                }
                
                with col2:
                    st.markdown("## Sentiment Analysis")
                    st.markdown(f"### Predicted Sentiment: <span style='color:{sentiment_colors.get(sentiment, 'blue')}'>{sentiment}</span>", unsafe_allow_html=True)
                    
                    # Display emoji based on sentiment
                    emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😟"
                    st.markdown(f"# {emoji}")
                    
                    # Display confidence scores as percentages
                    st.markdown("### Confidence Scores:")
                    for label, prob in zip(label_encoder.classes_, probabilities):
                        st.markdown(f"**{label}**: {prob*100:.2f}%")
                
                # Display probability chart
                st.markdown("### Probability Analysis")
                chart = plot_probability_chart(probabilities, label_encoder.classes_, optimal_thresholds)
                st.pyplot(chart)
                
                # Text analysis section
                st.markdown("### Text Analysis")
                col_text1, col_text2 = st.columns(2)
                
                with col_text1:
                    # Process text to show processed version
                    words_with_pos, processed_text = extract_words_with_pos(review_input)
                    
                    st.markdown("#### Key Words Identified:")
                    # Display important words with POS tags
                    important_words = [word for word, tag in words_with_pos 
                                      if word not in stop_words and len(word) > 2]
                    
                    # Display up to 15 important words as tags
                    if important_words:
                        word_html = ""
                        for word in important_words[:15]:
                            word_html += f'<span style="background-color: #e0e0e0; padding: 5px; margin: 5px; border-radius: 5px;">{word}</span>'
                        st.markdown(f'<div style="line-height: 2.5;">{word_html}</div>', unsafe_allow_html=True)
                    else:
                        st.write("No significant words identified.")
                
                with col_text2:
                    st.markdown("#### Processed Text:")
                    st.write(processed_text)
                
                # Display model performance metrics
                st.markdown("### Model Performance Metrics")
                st.write("The prediction uses class-specific thresholds to balance sensitivity and specificity:")
                display_threshold_metrics()
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                with debug_expander:
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()