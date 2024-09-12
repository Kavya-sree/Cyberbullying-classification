import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_single_text
from data_cleaning import clean_text

# Paths to the saved model, tokenizer, and label encoder
MODEL_PATH = "models/Bidirectional_LSTM_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

@st.cache_resource
def load_resources():
    """Load the model, tokenizer, and label encoder from disk."""
    try:
        # Load the model
        model = load_model(MODEL_PATH)
        
        # Load the tokenizer
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load the label encoder
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, tokenizer, label_encoder
    
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

def predict(model, tokenizer, label_encoder, text_input):
    """Preprocess the input text and predict its class."""
    # Clean the text
    cleaned_text = clean_text(text_input)

    # Preprocess the cleaned text
    preprocessed_text = preprocess_single_text(cleaned_text, tokenizer)
    print("Shape of preprocessed text:", preprocessed_text.shape)  # Debug line

    try:
        # Get prediction
        prediction = model.predict(preprocessed_text)  # Shape: (1, num_classes)
        print("Model prediction probabilities:", prediction)  # Debug line
        predicted_class = prediction.argmax(axis=1)     # Shape: (1,)
        
        # Map prediction to label
        label = label_encoder.inverse_transform(predicted_class)[0]
        return label
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

        return "Error"

def main():
    """Main function to run the Streamlit app."""
    # Load resources
    model, tokenizer, label_encoder = load_resources()
    
    st.title("Cyberbullying Classification")
    
    # Input text from user
    text_input = st.text_area("Enter a text to classify", "")
    
    if st.button("Predict"):
        if text_input:
            # Predict and display the result
            predicted_label = predict(model, tokenizer, label_encoder, text_input)
            st.write(f"Predicted Class: {predicted_label}")
        else:
            st.write("Please enter some text for classification.")

if __name__ == "__main__":
    main()
