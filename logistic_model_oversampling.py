from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import get_preprocessed_data
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# Get and preprocess the data
df = get_preprocessed_data()
X = df[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]]
y = df["Dementia"]

# Apply Random Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Logistic Regression with class weights
model = LogisticRegression(max_iter=1000)

# Perform 10-fold cross-validation with multiple metrics
scores = cross_validate(model, X_resampled, y_resampled, cv=10,
                        scoring={
                            'accuracy': 'accuracy',
                            'f1': 'f1',
                            'recall': 'recall',
                            'precision': 'precision'
                        },
                        return_train_score=False)

# Print results
print("Accuracy scores:", scores['test_accuracy'])
print("F1 scores:", scores['test_f1'])
print("Recall scores:", scores['test_recall'])
print("Precision scores:", scores['test_precision'])

print("\nMean Accuracy:", np.mean(scores['test_accuracy']))
print("Mean F1 Score:", np.mean(scores['test_f1']))
print("Mean Recall:", np.mean(scores['test_recall']))
print("Mean Precision:", np.mean(scores['test_precision']))

# Class distribution info
print("\nOriginal class distribution:\n", y.value_counts())
print("Resampled class distribution:\n", np.bincount(y_resampled))
