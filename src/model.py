#Se importan las clases para la crear un vectorizador TF-IDF, el clasificador basado en bosques aleatorios, un pipeline y la carga y descarga del modelo.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import dump #CAMBIADO DE from sklearn.externals import joblib

def build_model():
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier()),
    ])
    return model

def save_model(model, filepath):
    dump(model, filepath)
