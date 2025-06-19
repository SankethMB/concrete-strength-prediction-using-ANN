# 📦 Install required libraries if not done
# pip install tensorflow scikit-learn shap pandas numpy joblib

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import shap
import joblib
import tensorflow as tf

# 🗂️ Step 0: Create model folder if not exists
os.makedirs("model", exist_ok=True)

# 🧪 Step 1: Load Yeh’s dataset
url = "Concrete.csv"
df = pd.read_csv(url)

df.columns = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
              'CoarseAggregate', 'FineAggregate', 'Age', 'CompressiveStrength']

X = df.drop("CompressiveStrength", axis=1)
y = df["CompressiveStrength"]

# 🧼 Step 2: Normalize input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save the scaler under model/
joblib.dump(scaler, "model/scaler.pkl")

# 📊 Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🧠 Step 4: Build ANN model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# 🛑 Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 📈 Step 5: Train model
model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=300,
    batch_size=8,
    callbacks=[early_stop],
    verbose=0
)

# ✅ Step 6: Save trained model under model/
model.save("model/annmodel.keras")

# 🎯 Step 7: Evaluate
y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 ANN Model Evaluation:")
print(f"   R² Score : {r2:.4f}")
print(f"   MAE      : {mae:.2f} MPa")
print(f"   RMSE     : {rmse:.2f} MPa")

# 🧠 Step 8: SHAP value computation and save
explainer = shap.Explainer(model, X_train[:100])
shap_values = explainer(X_train[:100])

# ✅ Save SHAP values under model/
joblib.dump(shap_values, "model/shap_values.pkl")
