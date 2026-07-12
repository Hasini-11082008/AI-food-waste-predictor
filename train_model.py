import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset/food_waste_dataset.csv")

# Encode Food column
food_encoder = LabelEncoder()
data["Food"] = food_encoder.fit_transform(data["Food"])

# Encode Status column
status_encoder = LabelEncoder()
data["Status"] = status_encoder.fit_transform(data["Status"])

# Input and Output
X = data[["Food", "Quantity", "Storage_Days", "Temperature", "Expiry_Days"]]
y = data["Status"]

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save Model
joblib.dump(model, "model/waste_model.pkl")
joblib.dump(food_encoder, "model/food_encoder.pkl")
joblib.dump(status_encoder, "model/status_encoder.pkl")

print("✅ Model Trained Successfully!")