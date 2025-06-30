import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("oasis_cross-sectional.csv")


df = df.drop(columns=["ID", "Hand", "Delay"])

df = df[df["CDR"].notna()]
df["Dementia"] = df["CDR"].apply(lambda x: 1 if x> 0 else 0)
for col in ["MMSE", "SES", "Educ"]:
    df[col] = df[col].fillna(df[col].median())

df["M/F"] = df["M/F"].map({"M":1, "F": 0})

features_to_scale = ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print(df)