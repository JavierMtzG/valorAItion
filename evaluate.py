import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.preprocess import preprocess_data, load_data
from predict import load_model
from utilities.text_utilities import load_stop_words
import time

def analyze_results(accuracy, conf_matrix, class_report):
    print("\nAnalyzing results...\n")
    
    if accuracy > 0.8:
        print(f"Good accuracy: {accuracy:.2f}. The model is performing well.")
    elif accuracy > 0.6:
        print(f"Moderate accuracy: {accuracy:.2f}. The model is performing okay, but there's room for improvement.")
    else:
        print(f"Low accuracy: {accuracy:.2f}. The model needs significant improvement.")
    
    print("\nConfusion Matrix Analysis:")
    print(conf_matrix)
    print(f"True Negatives: {conf_matrix[0,0]}")
    print(f"False Positives: {conf_matrix[0,1]}")
    print(f"False Negatives: {conf_matrix[1,0]}")
    print(f"True Positives: {conf_matrix[1,1]}")
    
    print("\nClassification Report:")
    print(class_report)
    
    precision_neg = class_report['0']['precision']
    recall_neg = class_report['0']['recall']
    precision_pos = class_report['1']['precision']
    recall_pos = class_report['1']['recall']
    
    print(f"\nNegative class - Precision: {precision_neg:.2f}, Recall: {recall_neg:.2f}")
    print(f"Positive class - Precision: {precision_pos:.2f}, Recall: {recall_pos:.2f}")
    
    if precision_neg < 0.7 or recall_neg < 0.7:
        print("The model is not very good at identifying negative reviews. Consider improving the model or the data preprocessing.")
    if precision_pos < 0.7 or recall_pos < 0.7:
        print("The model is not very good at identifying positive reviews. Consider improving the model or the data preprocessing.")

def main():
    print("Loading stop words...")
    stop_words = load_stop_words('./utilities/stop_words_english.txt')
    
    print("Loading the model...")
    model = load_model('trained_model.pkl')
    
    print("Loading test data...")
    test_data = load_data('./data/test.csv')
    
    print("Preprocessing test data...")
    start_time = time.time()
    test_data = preprocess_data(test_data, stop_words)
    print(f"Test data preprocessed in {time.time() - start_time:.2f} seconds.")
    
    X_test = test_data['combined_text']
    y_true = test_data['clean_polarity']
    
    print("Making predictions on the test data...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    print(f"Predictions made in {time.time() - start_time:.2f} seconds.")
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix: ")
    print(conf_matrix)
    print("\nClassification Report: ")
    print(classification_report(y_true, y_pred))
    
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['True Neg', 'True Pos'], columns=['Pred Neg', 'Pred Pos'])
    conf_matrix_df.to_csv('confusion_matrix.csv')
    
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv('classification_report.csv')
    
    analyze_results(accuracy, conf_matrix, class_report)

if __name__ == "__main__":
    main()
