# Cyberbullying-Classification app

The Cyberbullying Detection App aims to identify and classify instances of cyberbullying in tweets. The app categorizes cyberbullying into the following specific categories:
1. Age
2. Ethnicity
3. Gender
4. Religion
5. Not Cyberbullying

## Table of Contents
- [Project Overview](#project-overview)
- [Challenges](#challenges)
- [Data](#data)
- [Key Features of model](#key-features-of-model)
- [Model Performance](#model-Performance)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)

## Project Overview

The Cyberbullying Classification project utilizes a Bi-Directional LSTM model to classify text data into different cyberbullying categories. The workflow involves data cleaning, preprocessing, model training, and evaluation. The model is implemented using TensorFlow and Keras.

## Challenges

When dealing with a class like not_cyberbullying that contains a wide range of topics and content types (e.g., reality shows, daily life, general comments), the challenge is that this class becomes very diverse and less cohesive compared to more specific classes (like ethnicity or gender in the context of cyberbullying). This diversity makes it difficult for the model to learn a consistent pattern, often resulting in lower recall and precision for the not_cyberbullying class.

## Data

The dataset used for this project is located in `data/` folder. The dataset is taken from (Kaggle)[https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification/data].
The original dataset contains 6 categories. I ommitted the category `other_cyberbullying` as it was too generic.

## Key Features of model

- Model Architecture: The Bi-LSTM architecture allows the model to capture contextual information from both directions in a sequence, improving the ability to detect nuances in cyberbullying language.
- Class Imbalance Handling: To address class imbalance, SMOTE (Synthetic Minority Oversampling Technique) is applied, ensuring the model doesn't favor more prevalent categories.
- Learning Rate Scheduler: A cyclical learning rate (CLR) scheduler is implemented to optimize the learning process.

## Model Performance

The model achieves the following performance metrics:

- Final Test Accuracy: 92.30%
- Final Test Loss: 0.2465

![Classification Report](imgs/classification_report.txt)


## Installation

To set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Kavya-sree/Cyberbullying-classification.git
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Training the Model**: To train the Bi-Directional LSTM model, simply run the `train.py` script:

```bash
python train.py
```
The following artifacts will be saved in the models/ directory:

- Trained Model: Bidirectional_LSTM_model.keras
- Tokenizer: tokenizer.pkl
- Label Encoder: label_encoder.pkl

2. **Streamlit App**: Launch the Streamlit app to interact with the model:

    ```bash
    streamlit run app.py
    ```

## Files

- `data/`: Contains the raw dataset `cyberbullying_data.csv`.
- `models/`: Directory where trained models, tokenizers, and label encoders are saved.
- `src/utils.py`: Contains utility functions such as cyclical learning rate scheduler.
- `src/data_cleaning.py`: Contains functions for cleaning the text data.
- `src/preprocessing.py`: Contains functions for preprocessing the text data.
- `src/model.py` : contain model architecture.
- `src/train.py`: Script to train the model and save the results.
- `app.py`: Streamlit application for model interaction.


## Future Improvements
* Expansion of Categories: Consider revising and adding more specific categories or subcategories based on user feedback and evolving needs.
* Enhanced Data: Incorporate additional features or external data sources to improve model accuracy and robustness.

