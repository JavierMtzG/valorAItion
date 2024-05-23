#El objetivo de este archivo es preprocesar los datos y preparar el conjunto para la formación del modelo
#Definimos distintas funciones para cargar el csv, y limpiar los comentarios y convertir las clasificaciones en polaridad 0 y 1
#utilizamos la función clean_text definida en el archivo text_utilities para poder limpiar las stop words, etc...

import pandas as pd
from utilities.text_utilities import clean_text

#La librería PANDAS tiene una función read_csv para leer este tipo de archivos
def load_data(filepath):
    data = pd.read_csv(filepath, names=['polarity','title','text'], quotechar="\"")
    return data

#en esta sencilla función, cargamos del path el archivo .csv, le indicamos las columnas existentes y los delimitadores 
#entre el título de la review y el texto (")

def preprocess_data(data):
    data['clean_title'] = data['title'].apply(clean_text)
    data['clean_text'] = data['text'].apply(clean_text)
    data['clean_polarity'] = data['polarity'] - 1
    return data
    
#En esta función, nos limitamos a limpiar los comentarios con la función definida en text_utilities