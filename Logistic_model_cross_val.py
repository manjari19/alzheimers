from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import get_preprocessed_data
import numpy as np

df = get_preprocessed_data()

X= df[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]]
y = df["Dementia"]

model = LogisticRegression(class_weight='balanced', max_iter=1000)


scores = cross_val_score(model, X, y, cv=10, scoring='f1')

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))