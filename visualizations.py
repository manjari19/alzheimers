import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import get_preprocessed_data

# Load preprocessed data
df = get_preprocessed_data()

# 1. Countplot: Number of Dementia vs. No Dementia
sns.countplot(x="Dementia", data=df)
plt.title("Count of Dementia Cases")
plt.xlabel("Dementia (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.show()

# 2. Histogram of MMSE scores by Dementia
sns.histplot(data=df, x="MMSE", hue="Dementia", bins=20, kde=True)
plt.title("Distribution of MMSE Scores by Dementia Status")
plt.xlabel("MMSE Score")
plt.ylabel("Count")
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Boxplot: Age vs Dementia
sns.boxplot(x="Dementia", y="Age", data=df)
plt.title("Age Distribution by Dementia Status")
plt.xlabel("Dementia (0 = No, 1 = Yes)")
plt.ylabel("Age (Standardized)")
plt.show()

# 5. Pairplot: Visualize feature interactions
selected_features = ["Age", "MMSE", "nWBV", "Dementia"]
sns.pairplot(df[selected_features], hue="Dementia", diag_kind="kde")
plt.suptitle("Pairwise Feature Comparison", y=1.02)
plt.show()
