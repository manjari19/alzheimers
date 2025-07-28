from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from preprocessing import get_preprocessed_data
import numpy as np

df = get_preprocessed_data()
X = df[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]]
y = df["Dementia"]

model = RandomForestClassifier(random_state=42)

# Cross-validation with multiple scoring metrics
scores = cross_validate(model, X, y, cv=10,
                        scoring={
                            'accuracy': 'accuracy',
                            'f1': 'f1',
                            'recall': 'recall',
                            'precision': 'precision'
                        },
                        return_train_score=False)

# Results
print("Accuracy scores:", scores['test_accuracy'])
print("F1 scores:", scores['test_f1'])
print("Recall scores:", scores['test_recall'])
print("Precision scores:", scores['test_precision'])


print("\nMean Accuracy:", np.mean(scores['test_accuracy']))
print("Mean F1 Score:", np.mean(scores['test_f1']))
print("Mean Recall:", np.mean(scores['test_recall']))
print("Mean Precision:", np.mean(scores['test_precision']))
