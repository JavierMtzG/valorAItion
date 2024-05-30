import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.preprocess import preprocess_data, load_stop_words

def load_model(filepath):
    print(f"Loading model from {filepath}...")
    model = joblib.load(filepath)
    print("Model loaded successfully.")
    return model

def main():
    print("Starting evaluation process...")
    stop_words = load_stop_words('./utilities/stop_words_english.txt')
    model = load_model('trained_model.pkl')
    
    print("Loading test data...")
    test_data = pd.read_csv('./data/test.csv', names=['polarity', 'title', 'text'], quotechar="\"", escapechar="\\", skiprows=1)
    test_data = preprocess_data(test_data, stop_words)
    
    X_test = test_data['combined_text']
    y_true = test_data['clean_polarity']
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("\nConfusion Matrix: ")
    print(conf_matrix)
    print("\nClassification Report: ")
    print(class_report)
    
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['True Neg', 'True Pos'], columns=['Pred Neg', 'Pred Pos'])
    conf_matrix_df.to_csv('confusion_matrix.csv')
    
    class_report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    class_report_df.to_csv('classification_report.csv')
    print("Evaluation process completed.")

if __name__ == "__main__":
    main()
