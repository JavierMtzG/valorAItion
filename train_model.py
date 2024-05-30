from src.preprocess import load_data, preprocess_data
from src.model import build_model, save_model
from utilities.text_utilities import load_stop_words
import time

def main():
    print("Loading stop words...")
    stop_words = load_stop_words('./utilities/stop_words_english.txt')
    
    print("Loading training data...")
    train_data = load_data("./data/train.csv")
    
    print("Preprocessing training data...")
    train_data = preprocess_data(train_data, stop_words)
    
    if train_data is None or train_data.empty:
        print("No data to train on. Exiting...")
        return
    
    # Muestreo de datos para pruebas rápidas
    sample_size = 5000  # Reducido para pruebas rápidas
    train_data = train_data.sample(n=sample_size, random_state=42)
    
    X_train = train_data['combined_text']
    y_train = train_data['clean_polarity']
    
    print("Building the model...")
    model = build_model(n_estimators=10, max_depth=5, max_features=10, n_jobs=2)  # Limitar a 2 núcleos
    
    print("Training the model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Model trained successfully in {time.time() - start_time:.2f} seconds.")
    
    print("Saving the trained model...")
    save_model(model, 'trained_model.pkl')

if __name__ == "__main__":
    main()
