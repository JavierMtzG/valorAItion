from src.preprocess import load_data, preprocess_data
from src.model import build_model, save_model
from utilities.text_utilities import load_stop_words

def main():
    stop_words = load_stop_words('./utilities/stop_words_english.txt')
    train_data = load_data("./data/train.csv")
    train_data = preprocess_data(train_data, stop_words)
    X_train = train_data['combined_text']
    y_train = train_data['clean_polarity']
    
    model = build_model()
    model.fit(X_train, y_train)
    
    save_model(model, 'trained_model.pkl')

if __name__ == "__main__":
    main()
