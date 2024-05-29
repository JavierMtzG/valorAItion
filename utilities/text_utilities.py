#este archivo define las utilidades que vayamos necesitando para el tratamiento del texto, estamos definiendo la función clean_text
#que va a tomar un texto de entrada y lo va a limpiar eliminando caracteres innecesarios
#La función a la que aludimos en la definición clean_text, re.sub, es parte del módulo re de python que sirve para trabajar con
#expresiones regulares, es una abreviatura de substitute, toma 3 argumentos, el primero que es el patron que se verificará en el texto
#Lo que añadiremos en lugar de las coincidencias que encuentre el patrón, y la cadena de texto donde queremos hacer las sustituciones
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def load_stop_words(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]
        return stop_words
    except Exception as e:
        print(f"Error loading stop words from {filepath}: {e}")
        return None

def clean_text(text, stop_words=None):
    text = str(text)
    
    # Remover comillas dobles internas y escapar nuevas líneas
    text = text.replace('""', '"').replace('\\n', '\n')
    
    # Limpiar el texto de caracteres no deseados
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


