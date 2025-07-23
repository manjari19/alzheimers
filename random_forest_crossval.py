from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import get_preprocessed_data
import numpy as np

df = get_preprocessed_data()
X = df[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]]
y = df["Dementia"]


model = RandomForestClassifier(random_state=42)


scores = cross_val_score(model, X, y, cv=10, scoring='f1')
print("Cross validation scores:", scores)
print("Mean accuracy:", np.mean(scores))


