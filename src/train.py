import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from utils import clr_triangular
from preprocessing import preprocess_data
from model import build_bidirectional_lstm_model
from data_cleaning import clean_dataset


def save_model_tokenizer_and_label_encoder(model, tokenizer, label_encoder, model_path=None, tokenizer_path=None, label_encoder_path=None):
    """
    Save model, tokenizer, and label encoder to disk.
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Save the label encoder
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label Encoder saved to {label_encoder_path}")

def create_directory(directory):
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def train_and_evaluate_model(X, y, model_builder, label_encoder, save_path=None, tokenizer_path=None, label_encoder_path=None, test_size=0.2, val_size=0.2):
    """
    Train and evaluate the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=np.argmax(y_train, axis=1))

    model = model_builder()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Cyclical learning rate callback
    clr_scheduler = LearningRateScheduler(clr_triangular)
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, clr_scheduler],
        verbose=2
    )

    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Score: loss of {scores[0]}; accuracy of {scores[1]*100}%')

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    
    # Display classification report with class names
    print(f'\nClassification Report:\n', classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    if save_path:
        create_directory(os.path.dirname(save_path))
        save_model_tokenizer_and_label_encoder(model, tokenizer, label_encoder, model_path=save_path, tokenizer_path=tokenizer_path, label_encoder_path=label_encoder_path)

if __name__ == "__main__":
    # Load and clean data
    df = pd.read_csv('data/cyberbullying_data.csv')  
    text_column = "tweet_text"
    df = clean_dataset(df, text_column)  
    X, y, label_encoder, tokenizer = preprocess_data(df)

    # Build model
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 for padding
    embedding_dim = 100  

    print("\nRunning Bi-LSTM model")
    save_model_path = "models/Bidirectional_LSTM_model.keras"
    save_tokenizer_path = "models/tokenizer.pkl"
    save_label_encoder_path = "models/label_encoder.pkl"

    train_and_evaluate_model(
        X, y, 
        lambda: build_bidirectional_lstm_model(vocab_size=vocab_size, embedding_dim=embedding_dim),
        label_encoder,
        save_path=save_model_path,
        tokenizer_path=save_tokenizer_path,
        label_encoder_path=save_label_encoder_path
    )

