#Se importan las clases para la crear un vectorizador TF-IDF, el clasificador basado en bosques aleatorios, un pipeline y la carga y descarga del modelo.
#SOLUCION 1 ANTE ERROR 1 noted en tfg, utilizar pipeline, como esperamos vectores en vez de texto crudo en el modelo, debemos usar un vectorizador TF-IDF.
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def build_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])
    return pipeline

def save_model(model, filepath):
    joblib.dump(model, filepath)
