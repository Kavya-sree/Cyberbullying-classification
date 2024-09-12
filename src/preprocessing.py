import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import tensorflow as tf

def preprocess_data(df, max_sequence_length=100):
    """
    Preprocess data for model training.
    - Tokenizes and pads sequences
    - Encodes labels
    - Applies SMOTE
    """
    X = df['text_cleaned'].values
    y = df['cyberbullying_type'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)  # Convert labels to one-hot encoding

    # Tokenization and Padding
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_sequence_length, padding='post')

    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    return X, y, label_encoder, tokenizer

def preprocess_single_text(text, tokenizer, max_sequence_length=100):
    """
    Preprocess a single text input for model prediction.
    """
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence to match the model's expected input length
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    
    return padded_sequence
