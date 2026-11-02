# predict.py
import os
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify, render_template, Response

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')
pca = joblib.load('pca.joblib')
model = joblib.load('knn_model.joblib')
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
encoder = joblib.load(os.path.join(BASE_DIR, 'encoder.joblib'))
pca = joblib.load(os.path.join(BASE_DIR, 'pca.joblib'))
model = joblib.load(os.path.join(BASE_DIR, 'knn_model.joblib'))

NUMERIC_FEATURES = ['Age','MonthlyIncome','DurationOfPitch','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting']
CAT_FEATURES = ['TypeofContact','Occupation','Gender','Designation','MaritalStatus','ProductPitched']

FIELD_MAPPING = {
    'age': 'Age',
    'monthlyIncome': 'MonthlyIncome',
    'pitchDuration': 'DurationOfPitch',
    'followups': 'NumberOfFollowups',
    'preferredStar': 'PreferredPropertyStar',
    'numberOfTrips': 'NumberOfTrips',
    'passport': 'Passport',
    'satisfaction': 'PitchSatisfactionScore',
    'ownCar': 'OwnCar',
    'numberOfChildren': 'NumberOfChildrenVisiting',
    'typeOfContact': 'TypeofContact',
    'occupation': 'Occupation',
    'gender': 'Gender',
    'designation': 'Designation',
    'maritalStatus': 'MaritalStatus',
    'productPitched': 'ProductPitched',
}

if PROMETHEUS_AVAILABLE:
    PREDICTION_COUNT = Counter(
        'holiday_prediction_requests_total',
        'Total number of holiday package prediction requests',
        ['outcome']
    )
    PREDICTION_LATENCY = Histogram(
        'holiday_prediction_latency_seconds',
        'Time spent processing holiday package prediction requests'
    )
    HTTP_REQUESTS_TOTAL = Counter(
        'holiday_http_requests_total',
        'Total HTTP requests received',
        ['method', 'endpoint', 'status']
    )

@app.after_request
def record_request_metrics(response):
    if PROMETHEUS_AVAILABLE and request.path != '/metrics':
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code
        ).inc()
    return response

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.route('/metrics')
def metrics():
    if not PROMETHEUS_AVAILABLE:
        return Response("# prometheus_client not installed\n", mimetype="text/plain")
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    start_time = time.time()
    raw_data = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not raw_data:
        raw_data = request.form.to_dict()

    # Normalize field names (support both camelCase from form.html and PascalCase)
    data = dict(raw_data)
    for src_key, dst_key in FIELD_MAPPING.items():
        if dst_key not in data and src_key in data:
            data[dst_key] = data[src_key]

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
    prediction = int(model.predict(X_pca)[0])
    probabilities = model.predict_proba(X_pca)[0]
    buy_probability = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)

    outcome_label = "buy" if prediction == 1 else "no_buy"
    if PROMETHEUS_AVAILABLE:
        PREDICTION_COUNT.labels(outcome=outcome_label).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

    result = "YES  - Likely to Buy" if prediction == 1 else "NO  - Not Likely to Buy"

    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({
            "prediction": prediction,
            "probability": buy_probability,
            "result": result
        })

    return render_template('form.html', result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
