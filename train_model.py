import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset (THIS LINE IS VERY IMPORTANT)
data = pd.read_csv("creditcard.csv")

# Select only required columns
X = data[["Time", "Amount"]]
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Model trained and saved successfully")