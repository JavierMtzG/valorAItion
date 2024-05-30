from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def build_model(n_estimators=10, max_depth=5, max_features=10, n_jobs=2):
    print("Building model with the following parameters:")
    print(f"n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}, n_jobs={n_jobs}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),  # Limitar el número de características de TF-IDF
        ('clf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=n_jobs))
    ])
    print("Model built successfully.")
    return pipeline

def save_model(model, filepath):
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    print("Model saved successfully.")
