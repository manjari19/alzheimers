from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import get_preprocessed_data

df = get_preprocessed_data()
X = df[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]]
y = df["Dementia"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test, y_pred))