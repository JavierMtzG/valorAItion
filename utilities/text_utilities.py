#este archivo define las utilidades que vayamos necesitando para el tratamiento del texto, estamos definiendo la función clean_text
#que va a tomar un texto de entrada y lo va a limpiar eliminando caracteres innecesarios
#La función a la que aludimos en la definición clean_text, re.sub, es parte del módulo re de python que sirve para trabajar con
#expresiones regulares, es una abreviatura de substitute, toma 3 argumentos, el primero que es el patron que se verificará en el texto
#Lo que añadiremos en lugar de las coincidencias que encuentre el patrón, y la cadena de texto donde queremos hacer las sustituciones
import re

def clean_text(text, stop_words=None):
    # Limpiar el texto eliminando caracteres no alfanuméricos y convirtiéndolo a minúsculas
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()

    # Eliminar las stop words si se proporcionan para mejorar la calidad del texto y reducir el ruido
    if stop_words:
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    
    return text