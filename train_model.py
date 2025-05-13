from src.preprocess import load_data, preprocess_data
from src.model import build_model, save_model
from utilities.text_utilities import load_stop_words
from sklearn.model_selection import GridSearchCV, train_test_split
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
    sample_size = 3600000  # Reducido para pruebas rápidas
    train_data = train_data.sample(n=sample_size, random_state=42)
    
    X_train = train_data['combined_text']
    y_train = train_data['clean_polarity']
    
    print("Construyendo el modelo base...")
    base_model = build_model()
    
    print("Aplicando Grid Search para optimización de hiperparámetros...")
    parametros_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [20, 30],
        'clf__max_features': [50, 100]
    }
    grid_search = GridSearchCV(base_model, parametros_grid, cv=3, scoring='accuracy', n_jobs=4, verbose=2)

    print("Entrenando el modelo...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print(f"Modelo entrenado satisfactoriamente en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    print("Guardando el modelo entrenado...")
    save_model(best_model, 'trained_model.pkl')

if __name__ == "__main__":
    main()
