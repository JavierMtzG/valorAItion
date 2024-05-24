from sklearn.externals import joblib
from utilities.text_utilities import clean_text

def load_model(filepath):
    return joblib.load(filepath)

def main():
    model = load_model('trained_model.pkl')
    title = input("Introduce el título de tu review: ")
    text = input("Introduce el texto de tu review: ")
    title = clean_text(title)
    text = clean_text(text)
    predicted_polarity = model.predict([[title, text]])[0]
    print(f"La polaridad la review introducida es {'positiva' if predicted_polarity == 1 else 'negativa'}")

if __name__ == "__main__":
    main()
