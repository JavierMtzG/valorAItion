from src.preprocess import load_data, preprocess_data 
from src.model import build_model, save_model

def main():
    train_data = load_data("./data/train.csv")
    train_data = preprocess_data(train_data)
    train_data['combined_text'] = train_data['clean_title'] + ' ' + train_data['clean_text']
    print(train_data.shape)  # Agregar esta línea para imprimir la forma de los datos antes de ajustar el modelo
    model = build_model()
    model.fit(train_data[['combined_text']], train_data['clean_polarity']) 
    save_model(model, 'trained_model.pkls')

if __name__ == "__main__":
    main()
