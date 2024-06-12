import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def load_stop_words(filepath):
    try:
        print(f"Cargando Stop words desde {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]
        print("Carga completa de las Stop Words.")
        return stop_words
    except Exception as e:
        print(f"Error cargando las Stop Words de {filepath}: {e}")
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
