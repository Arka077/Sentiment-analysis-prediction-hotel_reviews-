import streamlit as st
import numpy as np
import re
import joblib
import nltk
import tensorflow as tf
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download necessary resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

download_nltk_resources()

# Load model and preprocessing components
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("imdb_sentiment_model.h5")
    label_encoder = joblib.load("label_encoder.pkl")
    word_to_index = joblib.load("word_to_index.pkl")
    model_params = joblib.load("model_params.pkl")
    return model, label_encoder, word_to_index, model_params

model, label_encoder, word_to_index, model_params = load_model()
sent_length = model_params["sent_length"]
optimal_threshold = model_params.get("optimal_threshold")

# Set up text preprocessing
stop_words = set(stopwords.words('english'))
domain_stopwords = {'hotel', 'room', 'stay', 'night', 'day', 'days', 'stayed', 'property', 'resort', 'location', 'time'}
stop_words.update(domain_stopwords)
lemmatizer = WordNetLemmatizer()

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

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    # Clean text first
    text = clean_text(text)
    
    # Tokenize
    word_tokens = word_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(word_tokens)
    
    # Filter out stopwords and lemmatize based on POS
    tokens = []
    for word, tag in pos_tags:
        if word not in stop_words and len(word) > 2:  # Skip short words
            wordnet_pos = get_pos(tag)
            lemma = lemmatizer.lemmatize(word, wordnet_pos)
            tokens.append(lemma)
    
    return tokens

def texts_to_sequences(tokenized_texts, word_to_index, max_length):
    sequences = []
    for tokens in tokenized_texts:
        # Convert tokens to indices, using 0 if token not in vocabulary
        sequence = [word_to_index.get(token, 0) for token in tokens]
        sequences.append(sequence)
    
    # Pad sequences
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='pre'
    )
    return padded_sequences

def predict_sentiment(review_text):
    # Process the review
    tokens = preprocess_text(review_text)
    
    # Convert to sequence
    sequences = texts_to_sequences([tokens], word_to_index, sent_length)
    
    # Get prediction
    pred_prob = model.predict(sequences)
    
    # For binary classification
    if len(label_encoder.classes_) == 2:
        pred_prob = pred_prob.flatten()[0]
        pred_class = int(pred_prob >= optimal_threshold)
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        
        return pred_label, pred_prob
    else:
        # For multi-class
        pred_class = np.argmax(pred_prob, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        
        return pred_label, pred_prob[0]

# Streamlit app
st.title("Hotel Review Sentiment Analysis")

# User input
review_text = st.text_area("Enter your hotel review:", height=150)

# Process input
if st.button("Analyze Sentiment"):
    if not review_text:
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            sentiment, probability = predict_sentiment(review_text)
            
            # Display results with appropriate colors
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment}")
                color = "green"
            elif sentiment == "Negative":
                st.error(f"Sentiment: {sentiment}")
                color = "red"
            else:
                st.info(f"Sentiment: {sentiment}")
                color = "blue"
            
            # Display probability
            if len(label_encoder.classes_) == 2:
                st.write(f"Confidence: {probability:.2f}")
                
                # Simple progress bar for confidence
                st.progress(float(probability))
            else:
                # For multi-class, show all probabilities
                for i, cls in enumerate(label_encoder.classes_):
                    st.write(f"{cls}: {probability[i]:.2f}")