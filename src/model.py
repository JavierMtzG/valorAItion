#Se importan las clases para la crear un vectorizador TF-IDF, el clasificador basado en bosques aleatorios, un pipeline y la carga y descarga del modelo.
#SOLUCION 1 ANTE ERROR 1 noted en tfg, utilizar pipeline, como esperamos vectores en vez de texto crudo en el modelo, debemos usar un vectorizador TF-IDF.
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib # type: ignore

def build_model(): #~modificacion vs v1 es añadir el pipeline para facilitar la union entre la vectorizacion y el random forest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])
    return pipeline

def save_model(model, filepath):
    joblib.dump(model, filepath) #dump del modelo para save el pkl