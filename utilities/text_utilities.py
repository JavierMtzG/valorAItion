import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def load_stop_words(filepath):
    try:
        print(f"Loading stop words from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]
        print("Stop words loaded successfully.")
        return stop_words
    except Exception as e:
        print(f"Error loading stop words from {filepath}: {e}")
        return None

def clean_text(text, stop_words=None):
    text = str(text)
    text = text.replace('""', '"').replace('\\n', '\n')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    
    tokens = text.split()
    
    if stop_words:
        stop_words_set = set(stop_words)
    else:
        stop_words_set = set(stopwords.words('english'))
    
    tokens = [token for token in tokens if token not in stop_words_set]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)
