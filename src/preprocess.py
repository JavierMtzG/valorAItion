#El objetivo de este archivo es preprocesar los datos y preparar el conjunto para la formación del modelo
#Definimos distintas funciones para cargar el csv, y limpiar los comentarios y convertir las clasificaciones en polaridad 0 y 1
#utilizamos la función clean_text definida en el archivo text_utilities para poder limpiar las stop words, etc...
import csv
import pandas as pd
from utilities.text_utilities import clean_text

def load_data(filepath):
    try:
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la primera fila (encabezados)
            for row in reader:
                data.append(row)
        return pd.DataFrame(data, columns=['polarity', 'title', 'text'])
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def preprocess_data(data, stop_words=None):
    if data is None:
        return None
    
    # Limpiar el título y el texto antes de la combinación
    data['clean_title'] = data['title'].apply(lambda x: clean_text(x, stop_words=stop_words))
    data['clean_text'] = data['text'].apply(lambda x: clean_text(x, stop_words=stop_words))
    
    # Combinar el título y el texto limpios
    data['combined_text'] = data['clean_title'] + ' ' + data['clean_text']
    
    # Mapear la polaridad a valores numéricos
    data['clean_polarity'] = data['polarity'].apply(lambda x: 0 if x == '1' else 1)
    
    return data


#En esta función, nos limitamos a limpiar los comentarios con la función definida en text_utilities