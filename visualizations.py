import seaborn as sns
import matplotlib.pyplot as plt
import os
from preprocessing import get_preprocessed_data

# Load preprocessed data
df = get_preprocessed_data()

# Create folder if it doesn't exist
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# 1. Countplot: Number of Dementia vs. No Dementia
plt.figure()
sns.countplot(x="Dementia", data=df)
plt.title("Count of Dementia Cases")
plt.xlabel("Dementia (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.savefig(f"{output_dir}/dementia_countplot.png")
plt.close()

# 2. Histogram of MMSE scores by Dementia
plt.figure()
sns.histplot(data=df, x="MMSE", hue="Dementia", bins=20, kde=True)
plt.title("Distribution of MMSE Scores by Dementia Status")
plt.xlabel("MMSE Score")
plt.ylabel("Count")
plt.savefig(f"{output_dir}/mmse_histogram.png")
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# 4. Boxplot: Age vs Dementia
plt.figure()
sns.boxplot(x="Dementia", y="Age", data=df)
plt.title("Age Distribution by Dementia Status")
plt.xlabel("Dementia (0 = No, 1 = Yes)")
plt.ylabel("Age (Standardized)")
plt.savefig(f"{output_dir}/age_boxplot.png")
plt.close()

# 5. Pairplot: Visualize feature interactions
selected_features = ["Age", "MMSE", "nWBV", "Dementia"]
pairplot = sns.pairplot(df[selected_features], hue="Dementia", diag_kind="kde")
pairplot.fig.suptitle("Pairwise Feature Comparison", y=1.02)
pairplot.savefig(f"{output_dir}/pairplot_features.png")
plt.close()
