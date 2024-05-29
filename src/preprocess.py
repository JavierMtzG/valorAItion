#El objetivo de este archivo es preprocesar los datos y preparar el conjunto para la formación del modelo
#Definimos distintas funciones para cargar el csv, y limpiar los comentarios y convertir las clasificaciones en polaridad 0 y 1
#utilizamos la función clean_text definida en el archivo text_utilities para poder limpiar las stop words, etc...
import pandas as pd
import csv
from utilities.text_utilities import clean_text

def preprocess_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        # Aquí puedes aplicar cualquier lógica de limpieza que necesites
        processed_line = line.replace('""', '"').replace('\\"', '"')
        processed_lines.append(processed_line)

    with open('processed_train.csv', 'w', encoding='utf-8') as file:
        file.writelines(processed_lines)

    return 'processed_train.csv'

def load_data(filepath):
    processed_filepath = preprocess_csv(filepath)
    data = pd.read_csv(processed_filepath, 
                       names=['polarity', 'title', 'text'], 
                       quotechar='"', 
                       escapechar='\\', 
                       skiprows=1, 
                       quoting=csv.QUOTE_ALL, 
                       engine='python', 
                       error_bad_lines=False)
    return data

def preprocess_data(data, stop_words=None):
    if data is None:
        return None
    data['clean_title'] = data['title'].apply(lambda x: clean_text(x, stop_words=stop_words))
    data['clean_text'] = data['text'].apply(lambda x: clean_text(x, stop_words=stop_words))
    data['combined_text'] = data['clean_title'] + ' ' + data['clean_text']
    data['clean_polarity'] = data['polarity'].apply(lambda x: 0 if x == 1 else 1)
    return data

#En esta función, nos limitamos a limpiar los comentarios con la función definida en text_utilities