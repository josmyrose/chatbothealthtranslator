

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("symptoms.csv")
disease_counts = df["diseases"].value_counts()
valid_diseases = disease_counts[disease_counts > 1].index
df = df[df["diseases"].isin(valid_diseases)]

X = df.drop("diseases", axis=1)
y = df["diseases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Save model and symptom list
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("symptom_list.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
