import csv
import pandas as pd 
from utilities.text_utilities import clean_text

def load_data(filepath):
    try:
        print(f"Cargando datos de {filepath}...") 
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            #next(reader)  # Descomentar si en la primera fila aparecen encabezados
            for row in reader:
                data.append(row)
        print("Datos cargados exitosamente.")
        return pd.DataFrame(data, columns=['polarity', 'title', 'text'])
    except Exception as e:
        print(f"Error al cargar datos desde {filepath}: {e}")
        return None

def preprocess_data(data, stop_words=None):
    if data is None:
        return None
    
    print("Limpiando título y texto...")
    data['clean_title'] = data['title'].apply(lambda x: clean_text(x, stop_words=stop_words))
    data['clean_text'] = data['text'].apply(lambda x: clean_text(x, stop_words=stop_words))
    
    print("Combinando título y texto limpios...")
    data['combined_text'] = data['clean_title'] + ' ' + data['clean_text']
    
    print("Mapeando polaridad a valores numéricos...")
    data['clean_polarity'] = data['polarity'].apply(lambda x: 0 if x == '1' else 1)
    
    print("Preprocesamiento terminado.")
    return data
