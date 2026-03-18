# predict.py
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load models
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')
pca = joblib.load('pca.joblib')
model = joblib.load('knn_model.joblib')

NUMERIC_FEATURES = ['Age','MonthlyIncome','DurationOfPitch','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting']
CAT_FEATURES = ['TypeofContact','Occupation','Gender','Designation','MaritalStatus','ProductPitched']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Convert numeric values
    for col in NUMERIC_FEATURES:
        data[col] = float(data[col])

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Preprocessing
    X_num_scaled = scaler.transform(df[NUMERIC_FEATURES])
    X_cat_encoded = encoder.transform(df[CAT_FEATURES])
    X_processed = np.concatenate([X_cat_encoded, X_num_scaled], axis=1)
    X_pca = pca.transform(X_processed)

    # Predict
    prediction = model.predict(X_pca)[0]

    result = "YES  - Likely to Buy" if prediction == 1 else "NO  - Not Likely to Buy"
    return render_template('form.html', result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
