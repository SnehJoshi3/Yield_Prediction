# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive

# 2. Load dataset
file_path = '/content/drive/MyDrive/Colab Notebooks/Crop Yield Prediction for AgriBoost India.csv'
df = pd.read_csv(file_path)
df_encoded = pd.get_dummies(df, drop_first=True)
print(df.head())

# 3. Split into X and y
X=df_encoded.drop(columns='predicted_yield_quintal_per_hectare')
y = df['predicted_yield_quintal_per_hectare']

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# 5. Create Linear Regression model
model = LinearRegression()

# 6. Train the model
model.fit(X_train, y_train)
     
# 7. Predict using the model
y_pred = model.predict(X_test)

# 8. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Example Input
new_input = pd.DataFrame([{
    'crop_type': 'Wheat',
    'soil_type': 'Loamy',
    'rainfall_mm': 150.0,
    'temperature_c': 25.0,
    'humidity_percent': 60.0,
    'soil_ph': 6.5,
    'fertilizer_used_kg_per_hectare': 100.0,
    'pesticide_used_kg_per_hectare': 5.0,
    'irrigation_type': 'Drip',
    'seed_quality_index': 0.85,
    'satellite_ndvi_index': 0.72
}])

new_input_encoded = pd.get_dummies(new_input)
new_input_encoded = new_input_encoded.reindex(columns=X.columns, fill_value=0)
predicted_yield = model.predict(new_input_encoded)
print("Predicted Yield:", predicted_yield[0])
