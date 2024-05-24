#este archivo define las utilidades que vayamos necesitando para el tratamiento del texto, estamos definiendo la función clean_text
#que va a tomar un texto de entrada y lo va a limpiar eliminando caracteres innecesarios
#La función a la que aludimos en la definición clean_text, re.sub, es parte del módulo re de python que sirve para trabajar con
#expresiones regulares, es una abreviatura de substitute, toma 3 argumentos, el primero que es el patron que se verificará en el texto
#Lo que añadiremos en lugar de las coincidencias que encuentre el patrón, y la cadena de texto donde queremos hacer las sustituciones
import re

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    return text