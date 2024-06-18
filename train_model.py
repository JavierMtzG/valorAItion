from src.preprocess import load_data, preprocess_data
from src.model import build_model, save_model
from utilities.text_utilities import load_stop_words
import time

def main():
    print("Cargando las Stop Words...")
    stop_words = load_stop_words('./utilities/stop_words_english.txt')
    
    print("Cargando los datos de entrenamiento...")
    train_data = load_data("./data/train.csv")
    start_time_preprocess = time.time()
    print("Preprocesando los datos de entrenamiento...")
    train_data = preprocess_data(train_data, stop_words)
    print(f"Preprocesamiento completado en {time.time() - start_time_preprocess:.2f} segundos.")

    if train_data is None or train_data.empty:
        print("No existen datos para entrenar. Saliendo...")
        return
    
    # Muestreo de datos para pruebas rápidas
    sample_size = 1800000  # Reducido para pruebas rápidas
    train_data = train_data.sample(n=sample_size, random_state=42)
    
    X_train = train_data['combined_text']
    y_train = train_data['clean_polarity']
    
    print("Construyendo el modelo...")
    model = build_model(n_estimators=300, max_depth=30, max_features=50, n_jobs=6)  # Ajustes aumentados
    
    print("Entrenando el modelo...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Modelo entrenado satisfactoriamente en {time.time() - start_time:.2f} segundos.")
    
    print("Guardando el modelo entrenado...")
    save_model(model, 'trained_model.pkl')

if __name__ == "__main__":
    main()
