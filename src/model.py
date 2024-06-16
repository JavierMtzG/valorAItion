from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def build_model(n_estimators=300, max_depth=30, max_features=50, n_jobs=6):
    print("Construyendo el modelo con los siguientes parámetros:")
    print(f"n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}, n_jobs={n_jobs}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000)),  # Aumentar el número de características
        ('clf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=n_jobs, class_weight='balanced'))
    ])
    print("Modelo construido satisfactoriamente.")
    return pipeline

def save_model(model, filepath):
    print(f"Guardando modelo en {filepath}...")
    joblib.dump(model, filepath)
    print("Model guardado satisfactoriamente.")
