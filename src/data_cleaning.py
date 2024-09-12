import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Contractions dictionary
CONTRACTIONS_DICT = {
    "can't": "can not",
    "n't": " not",  # catches don't -> do not, isn't -> is not, etc.
    "n’t": " not", # (curly apostrophe)
    "'re": " are",  # catches you're -> you are, they're -> they are, etc.
    "’re": " are", # (curly apostrophe)
    "'s": " is",  # catches she's -> she is, he's -> he is, etc.
    "’s": " is", # (curly apostrophe)
    "'d": " would",  # catches I'd -> I would, you'd -> you would, etc.
    "’d": " would", # I’d -> I would (curly apostrophe)
    "'ll": " will",  # catches I'll -> I will, you'll -> you will, etc.
    "’ll": " will", # I’ll -> I will (curly apostrophe)
    "'ve": " have",  # catches I've -> I have, you've -> you have, etc.
    "’ve": " have", # I’ve -> I have (curly apostrophe)
    "'m": " am",  # catches I'm -> I am
    "’m": " am",    # I’m -> I am (curly apostrophe)
    "y'all": " you all", # y'all -> you all
    "y’all": "you all"  # y’all -> you all (curly apostrophe)
}

def expand_contractions(text, contractions_dict=CONTRACTIONS_DICT):
    """
    Expand contractions in the text using the provided contractions dictionary.
    
    Parameters:
    text (str): The input text.
    contractions_dict (dict): Dictionary of contractions and their expanded forms.
    
    Returns:
    str: Text with expanded contractions.
    """
    # Replace contractions using regex to identify patterns in the text
    for contraction, expansion in contractions_dict.items():
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        text = pattern.sub(expansion, text)
    
    return text

SLANG_MISSPELLED_DICT = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "gr8": "great",
    "luv": "love",
    "plz": "please",
    "thx": "thanks",
    "b4": "before",
    "thru": "through",
    "wat": "what",
    "cuz": "because",
    "bcuz": "because",
    "btw": "by the way",
    "gonna": "going to",
    "wanna": "want to",
    "l8": "late",
    "ne1": "anyone",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "sry": "sorry",
    "ttyl": "talk to you later",
    "smh": "shaking my head",
    "c u": "see you",
    "yr": "your",
    "k": "ok",
    "nvm": "never mind",
    "gd": "good",
    "txt": "text",
    "frm": "from",
    "gd nite": "good night",
    "w/": "with",
    "lmao": "laughing my ass off",
    "prolly": "probably",
    "fav": "favorite",
    "fam": "family",
    "lit": "exciting",
    "dont": "do not",
    "didnt": "did not",
    "cant": "can not"
}

def correct_misspellings(text, slang_misspelled_dict= SLANG_MISSPELLED_DICT):
    """
    Correct common misspelled words in a given text using a predefined dictionary.
    
    Args:
    text (str): The input text to be corrected.
    misspelled_dict (dict): A dictionary of common misspellings and their corrections.

    Returns:
    str: The corrected text.
    """
    # Create a regex pattern for each word to match, considering word boundaries
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang_misspelled_dict.keys()) + r')\b', re.IGNORECASE)

    # Replace each slang/misspelled word with its corrected form
    corrected_text = pattern.sub(lambda x: slang_misspelled_dict[x.group().lower()], text)
    return corrected_text

def clean_hashtags(text):
    # Remove hashtags at the end of the sentence
    text = re.sub(r'(\s+#[\w-]+)+\s*$', '', text).strip()
    
    # Remove the # symbol from hashtags in the middle of the sentence
    text = re.sub(r'#([\w-]+)', r'\1', text).strip()
    
    return text

# Remove accented character
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_patterns(text):
    """
    Remove various unwanted patterns from the text such as emails, URLs, mentions, etc.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    str: Text with removed patterns.
    """

    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Removing RT and QT
    text = re.sub(r'(\brt\b|\bqt\b)', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    def clean_hashtags(text):
        # Remove hashtags at the end of the sentence
        text = re.sub(r'(\s+#[\w-]+)+\s*$', '', text).strip()
    
        # Remove the # symbol from hashtags in the middle of the sentence
        text = re.sub(r'#([\w-]+)', r'\1', text).strip()
    
        return text
    
    text = clean_hashtags(text)

    # Remove Unicode-based emojis
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, '', text)

    # Remove traditional keyboard-based emoticons
    emoticon_pattern = r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"
    text = re.sub(emoticon_pattern, '', text)

    # Reduce elongated characters (e.g., "cooool" -> "cool")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Replace ".." or "..." with a space
    text = re.sub(r'\.{2,}', ' ', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove spaces between single characters
    text = re.sub(r'(?<=\b\w)\s(?=\w\b)', '', text)

    # Remove extra whitespace
    text = text.strip()
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from the text.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    str: Text without stopwords.
    """
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenize and lemmatize the text.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    str: Lemmatized text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)



def clean_text(text):
    """
    Perform all cleaning steps on the tweet text.
    
    Parameters:
    text (str): The tweet text to be cleaned.
    
    Returns:
    str: The cleaned tweet text.
    """
    text = text.lower()
    text = expand_contractions(text)
    text = correct_misspellings(text)
    text = remove_accented_chars(text)
    text = remove_patterns(text)
    text = remove_stopwords(text)
    text = tokenize_and_lemmatize(text)
    return text

def clean_dataset(df, text_column):
    """Apply cleaning to a DataFrame."""
    df['text_cleaned'] = df[text_column].apply(clean_text)
    return df
