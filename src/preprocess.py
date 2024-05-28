#El objetivo de este archivo es preprocesar los datos y preparar el conjunto para la formación del modelo
#Definimos distintas funciones para cargar el csv, y limpiar los comentarios y convertir las clasificaciones en polaridad 0 y 1
#utilizamos la función clean_text definida en el archivo text_utilities para poder limpiar las stop words, etc...
import csv #nueva incorporación para leer el archivo csv
import pandas as pd # type: ignore
from utilities.text_utilities import clean_text


def load_stop_words(filepath): #función para cargar las stop words
    try: #intentamos cargar las stop words
        with open(filepath, 'r', encoding='utf-8') as file: #abrimos el archivo
            stop_words = [line.strip() for line in file] #leemos las stop words
        return stop_words
    except Exception as e:
        print(f"Error loading stop words from {filepath}: {e}")
        return None

def load_data(filepath):
    data = pd.read_csv(filepath, names=['polarity', 'title', 'text'], quotechar="\"")
    return data

def preprocess_data(data, stop_words=None):
    if data is None: #si no hay datos, retornamos None
        return None
    data['clean_title'] = data['title'].apply(clean_text, stop_words=stop_words) #limpiamos el título
    data['clean_text'] = data['text'].apply(clean_text, stop_words=stop_words) #limpiamos el texto
    data['clean_polarity'] = data['polarity'].apply(lambda x: 0 if x == 1 else 1) #convertimos la polaridad en 0 y 1
    return data
    
#En esta función, nos limitamos a limpiar los comentarios con la función definida en text_utilities