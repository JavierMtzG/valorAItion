import joblib
from utilities.text_utilities import clean_text, load_stop_words
from lime.lime_text import LimeTextExplainer
import numpy as np

def load_model(filepath):
    print(f"Loading model from {filepath}...")
    model = joblib.load(filepath)
    print("Model loaded successfully.")
    return model


def main():
    stop_words = load_stop_words('./utilities/stop_words_english.txt')
    model = load_model('trained_model.pkl')

    title = input("Introduce el título de tu review: ")
    text = input("Introduce el texto de tu review: ")

    title_clean = clean_text(title, stop_words)
    text_clean = clean_text(text, stop_words)

    combined = title_clean + ' ' + text_clean
    predicted_polarity = model.predict([combined])[0]

    print(f"La polaridad de la review introducida es {'positiva' if predicted_polarity == 1 else 'negativa'}")

    print("\nAplicando LIME para explicación local...")
    explainer = LimeTextExplainer(class_names=['negativa', 'positiva'])
    explanation = explainer.explain_instance(combined, model.predict_proba, num_features=10)
    print("\nPalabras más influyentes en la predicción:")
    for word, weight in explanation.as_list():
        print(f"{word}: {weight:.4f}")


if __name__ == "__main__":
    main()