import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Preprocessing
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")
print("Model saved to model/model.pkl")
