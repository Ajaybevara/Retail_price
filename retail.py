# retail_price_optimization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Step 1: Load dataset
file_path = r"C:\Users\ajayb\OneDrive\Desktop\AIML_projects\Retail_Price\retail_price.csv"
df = pd.read_csv(file_path)

# Step 2: Preview data
print("Dataset Preview:\n", df.head())

# Step 3: Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Drop rows with missing values
df.dropna(inplace=True)

# Step 5: Drop non-numeric/irrelevant columns for modeling
df = df.drop(['product_id', 'month_year', 'product_category_name'], axis=1)

# Step 6: Show remaining columns
print("\nRemaining Columns:\n", df.columns)

# Step 7: Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 8: Define features (X) and target (y)
X = df.drop('unit_price', axis=1)
y = df['unit_price']

# Step 9: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Step 11: Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 12: Evaluation Function (compatible with old sklearn)
def evaluate_model(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # manual square root
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Step 13: Evaluate Models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest Regressor", y_test, y_pred_rf)

# Step 14: Plot Actual vs Predicted for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Random Forest)")
plt.grid(True)
plt.show()

# Step 15: Feature Importance (Random Forest)
feature_importance = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Step 16: Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("\n✅ Model saved as 'rf_model.pkl'")
