from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def build_model():
    print("Building model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=50, n_jobs=-1))  # Reduce the number of trees for faster training
    ])
    print("Model built successfully.")
    return pipeline

def save_model(model, filepath):
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    print("Model saved successfully.")
