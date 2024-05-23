from src.data_preprocess import load_data, preprocess_data
from src.model import build_model, save_model

def main():
    train_data = load_data("./data/train.csv")
    train_data = preprocess_data(train_data)
    model = build_model()
    model.fit(train_data[['clean_title', 'clean_text']], train_data['clean_polarity'])
    save_model(model, 'trained_model.pkl')

if __name__ == "__main__":
    main()
