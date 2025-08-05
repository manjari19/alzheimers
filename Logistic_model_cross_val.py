from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import get_preprocessed_data
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


df = get_preprocessed_data()

X = df[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]]
y = df["Dementia"]

# Logistic Regression with balanced class weights
#model = LogisticRegression(class_weight='balanced', max_iter=1000)
model = LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=1000)

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

#Cross-Validation Metrics Boxplot
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

scores_dict = {
    'Accuracy': scores['test_accuracy'],
    'F1': scores['test_f1'],
    'Recall': scores['test_recall'],
    'Precision': scores['test_precision']
}

plt.figure(figsize=(8, 6))
plt.boxplot(scores_dict.values(), labels=scores_dict.keys())
plt.title("Cross-Validation Performance Metrics (Logistic Regression)")
plt.ylabel("Score")
plt.savefig(f"{output_dir}/cv_metrics_boxplot.png")
plt.show()
plt.close()

# Random Forest with same CV setup
rf_model = RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42)

rf_scores = cross_validate(rf_model, X, y, cv=10,
                           scoring={
                               'accuracy': 'accuracy',
                               'f1': 'f1',
                               'recall': 'recall',
                               'precision': 'precision'
                           },
                           return_train_score=False)

# Calculate mean scores
logreg_means = [np.mean(scores['test_accuracy']), np.mean(scores['test_f1']),
                np.mean(scores['test_recall']), np.mean(scores['test_precision'])]

rf_means = [np.mean(rf_scores['test_accuracy']), np.mean(rf_scores['test_f1']),
            np.mean(rf_scores['test_recall']), np.mean(rf_scores['test_precision'])]

# Comparison Bar Plot
metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, logreg_means, width, label='Logistic Regression')
plt.bar(x + width/2, rf_means, width, label='Random Forest')
plt.ylabel('Score')
plt.title('Model Comparison (10-Fold CV)')
plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/model_comparison.png")
plt.show()
plt.close()


# --- Confusion Matrix and ROC Curve (single train/test split) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.savefig(f"{output_dir}/confusion_matrix_logreg.png")
plt.show()
plt.close()

# ROC Curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/roc_curve_logreg.png")
plt.show()
plt.close()