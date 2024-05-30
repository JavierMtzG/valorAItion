import joblib
from utilities.text_utilities import clean_text
from src.preprocess import load_stop_words

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
    
    title = clean_text(title, stop_words)
    text = clean_text(text, stop_words)
    
    predicted_polarity = model.predict([title + ' ' + text])[0]
    
    print(f"La polaridad de la review introducida es {'positiva' if predicted_polarity == 1 else 'negativa'}")

if __name__ == "__main__":
    main()
