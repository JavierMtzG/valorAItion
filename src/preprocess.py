import csv
import pandas as pd
from utilities.text_utilities import clean_text

def load_data(filepath):
    try:
        print(f"Loading data from {filepath}...")
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la primera fila (encabezados)
            for row in reader:
                data.append(row)
        print("Data loaded successfully.")
        return pd.DataFrame(data, columns=['polarity', 'title', 'text'])
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def preprocess_data(data, stop_words=None):
    if data is None:
        return None
    
    print("Cleaning title and text...")
    data['clean_title'] = data['title'].apply(lambda x: clean_text(x, stop_words=stop_words))
    data['clean_text'] = data['text'].apply(lambda x: clean_text(x, stop_words=stop_words))
    
    print("Combining clean title and text...")
    data['combined_text'] = data['clean_title'] + ' ' + data['clean_text']
    
    print("Mapping polarity to numeric values...")
    data['clean_polarity'] = data['polarity'].apply(lambda x: 0 if x == '1' else 1)
    
    print("Preprocessing done.")
    return data
