import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the dataset
df = pd.read_csv('tokopedia_score_content_cleaned.csv')

# Replace NaN in content with empty string
df['content'] = df['content'].fillna('')

# Create binary labels (0 for negative, 1 for positive)
df['sentiment'] = df['score'].apply(lambda x: 1 if x >= 3 else 0)

# Prepare the data
X = df['content'].values
y = df['sentiment'].values

print(f"Total samples: {len(X)}")
print(f"Positive samples: {sum(y == 1)}")
print(f"Negative samples: {sum(y == 0)}\n")

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Initialize K-Fold Cross Validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
fold_metrics = []

# Perform K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(k_fold.split(X), 1):
    print(f"\n{'='*50}")
    print(f"Fold {fold}")
    print('='*50)
    
    # Split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # TF-IDF transformation
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    # Initialize and train SVM
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_tfidf, y_train)
    
    # Make predictions
    predictions = svm.predict(X_val_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    conf_matrix = confusion_matrix(y_val, predictions)
    
    # Store metrics
    fold_metrics.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # Print detailed evaluation for this fold
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print("                 Predicted Negative  Predicted Positive")
    print(f"Actual Negative       {conf_matrix[0][0]}                {conf_matrix[0][1]}")
    print(f"Actual Positive       {conf_matrix[1][0]}                {conf_matrix[1][1]}")

# Calculate and print average metrics
print("\n" + "="*50)
print("Average Metrics Across All Folds:")
print("="*50)
print(f"Accuracy: {np.mean([m['accuracy'] for m in fold_metrics]):.4f} (±{np.std([m['accuracy'] for m in fold_metrics]):.4f})")
print(f"Precision: {np.mean([m['precision'] for m in fold_metrics]):.4f} (±{np.std([m['precision'] for m in fold_metrics]):.4f})")
print(f"Recall: {np.mean([m['recall'] for m in fold_metrics]):.4f} (±{np.std([m['recall'] for m in fold_metrics]):.4f})")
print(f"F1-score: {np.mean([m['f1'] for m in fold_metrics]):.4f} (±{np.std([m['f1'] for m in fold_metrics]):.4f})")
