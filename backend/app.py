from flask import Flask, render_template, request
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load scaler and model
scaler = joblib.load('model/scaler.pkl')
ann_model = load_model('model/ann_model.keras')

# Helper function to format input
def preprocess_input(data):
    return np.array([
        data['Cement'],
        data['Blast'],
        data['Fly'],
        data['Water'],
        data['Superplasticizer'],
        data['Coarse'],
        data['Fine Aggregate'],
        data['Age']
    ])

# Route: Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route: About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route: Predictor Page (Form)
@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

# Route: Handle Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Cement': float(request.form['cement']),
            'Blast': float(request.form['blast']),
            'Fly': float(request.form['fly']),
            'Water': float(request.form['water']),
            'Superplasticizer': float(request.form['superplasticizer']),
            'Coarse': float(request.form['coarse']),
            'Fine Aggregate': float(request.form['fine']),
            'Age': int(request.form['age'])
        }

        input_data = preprocess_input(data)
        input_scaled = scaler.transform([input_data])  # Must be 2D

        # ANN model prediction
        ann_pred = round(float(ann_model.predict(input_scaled, verbose=0)[0][0]), 2)

        # SHAP explanation
        explainer = shap.Explainer(lambda x: ann_model.predict(x, verbose=0), input_scaled)
        shap_values = explainer(input_scaled)

        # Save SHAP plot
        shap_path = os.path.join('static', 'shap_plot.png')
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(shap_path)
        plt.close()

        # Return to predictor page with prediction result and SHAP image
        return render_template('predictor.html', ann_pred=ann_pred, shap_img=shap_path)

    except Exception as e:
        return render_template('predictor.html', ann_pred=f"Error: {str(e)}", shap_img=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
