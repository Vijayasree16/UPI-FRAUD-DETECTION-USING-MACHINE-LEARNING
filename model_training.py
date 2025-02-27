import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
data = pd.read_csv("C:/Users/Vijay/OneDrive/Desktop/UPI FRAUD DETECTION/fraud detection.csv").sample(10000, random_state=42)

# Step 1: Select relevant features and target
features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
target = 'isFraud'

# Encode categorical 'type' column
data_encoded = pd.get_dummies(data[features], columns=['type'], drop_first=True)

# Add the target variable
data_encoded[target] = data[target]

# Step 2: Split the data
X = data_encoded.drop(columns=[target])
y = data_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Train a Random Forest Classifier with class weights
model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Save the model
with open('fraud_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the feature names
with open('model_columns.pkl', 'wb') as file:
    pickle.dump(X.columns, file)

print("Model training completed and saved.")
