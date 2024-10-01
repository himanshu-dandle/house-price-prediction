import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the cleaned training data
train_cleaned = pd.read_csv('../data/train_cleaned.csv')  

# Separate features and target variable
X = train_cleaned.drop('SalePrice', axis=1)
y = train_cleaned['SalePrice']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (Gradient Boosting Regressor)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the model's performance on the validation set
preds = gb_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print(f"Gradient Boosting RMSE on Validation Set: {rmse}")

# Save the trained model to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(gb_model, file)

print("Model saved successfully as model.pkl")
