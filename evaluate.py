import pandas as pd
from joblib import load #CAMBIADO from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.preprocess import load_data, preprocess_data


#carga del modelo
def load_model(filepath):
    return joblib.load(filepath)

def main():
    # Cargar el modelo entrenado
    model = load_model('trained_model.pkl')

    # Cargar los datos de prueba
    test_data = pd.read_csv('./data/test.csv')

    # Preprocesar los datos de prueba
    test_data = preprocess_data(test_data)

    # Realizar predicciones
    X_test = test_data[['clean_title', 'clean_text']]
    y_true = test_data['polarity']
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_true, y_pred) #calculamos la precision entre lo predicho y lo real
    conf_matrix = confusion_matrix(y_true, y_pred) #generamos la matriz de confusion
    class_report = classification_report(y_true, y_pred) #genera el reporte de clasificacion

    # Mostrar los resultados
    print(f"Precision: {accuracy}")
    print("\nMatriz de Confusion: ")
    print(conf_matrix)
    print("\nReporte de Clasificacion: ")
    print(class_report)

    # Guardar los resultados en archivos separados
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    results_df.to_csv('evaluation_results.csv', index=False)

    # Guardar la matriz de confusión
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['True Neg', 'True Pos'], columns=['Pred Neg', 'Pred Pos'])
    conf_matrix_df.to_csv('confusion_matrix.csv')

    # Guardar el informe de clasificación
    class_report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    class_report_df.to_csv('classification_report.csv')

if __name__ == "__main__":
    main()
