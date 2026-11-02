import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Travel Purchase Prediction | ML Analytics",
    page_icon="🌴",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUMERIC_FEATURES = [
    'Age', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting'
]
CAT_FEATURES = [
    'TypeofContact', 'Occupation', 'Gender', 'Designation',
    'MaritalStatus', 'ProductPitched'
]


@st.cache_resource
def load_or_train_models():
    scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
    encoder_path = os.path.join(BASE_DIR, 'encoder.joblib')
    pca_path = os.path.join(BASE_DIR, 'pca.joblib')
    model_path = os.path.join(BASE_DIR, 'knn_model.joblib')

    if all(os.path.exists(p) for p in [scaler_path, encoder_path, pca_path, model_path]):
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        pca = joblib.load(pca_path)
        model = joblib.load(model_path)
        return scaler, encoder, pca, model

    # Fallback: train from Travel.csv if joblib files are missing on cloud deploy
    csv_path = os.path.join(BASE_DIR, 'Travel.csv')
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

    X = df.drop(['ProdTaken', 'CustomerID'], axis=1)
    y = df['ProdTaken']

    scaler = StandardScaler()
    scaler.fit(X[NUMERIC_FEATURES])
    joblib.dump(scaler, scaler_path)

    oe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    oe.fit(X[CAT_FEATURES])
    joblib.dump(oe, encoder_path)

    X_num_scaled = scaler.transform(X[NUMERIC_FEATURES])
    X_cat_encoded = oe.transform(X[CAT_FEATURES])
    X_processed = np.concatenate([X_cat_encoded, X_num_scaled], axis=1)

    pca = PCA(n_components=0.9)
    pca.fit(X_processed)
    joblib.dump(pca, pca_path)

    X_pca_final = pca.transform(X_processed)
    knn = KNeighborsClassifier(n_neighbors=3)
    model = knn.fit(X_pca_final, y)
    joblib.dump(model, model_path)

    return scaler, encoder, pca, model


scaler, encoder, pca, model = load_or_train_models()

# Custom styling matching static/index.css and templates/form.html
st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 50%, #48cae4 100%);
        color: white;
        padding: 2.2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(0, 119, 182, 0.25);
    }
    .hero-box h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }
    .hero-box p {
        margin: 0;
        font-size: 1.05rem;
        opacity: 0.95;
    }
    .result-yes {
        background: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 1.25rem;
        border-radius: 10px;
        color: #065f46;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .result-no {
        background: #fffbeb;
        border-left: 6px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 10px;
        color: #92400e;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    </style>
    <div class="hero-box">
        <h1>🌴 Travel Purchase Prediction</h1>
        <p>Predict customer travel package purchase probability using advanced analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.subheader("✈️ Personal Information")
    st.caption("Basic demographic details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age *", min_value=18, max_value=100, value=35, step=1)
        marital_status = st.selectbox("Marital Status *", ["Married", "Single", "Divorced"])
    with col2:
        gender = st.selectbox("Gender *", ["Male", "Female"])
        num_children = st.number_input("Number of Children *", min_value=0, max_value=10, value=1, step=1)

    st.divider()
    st.subheader("💼 Professional & Financial")
    st.caption("Income and career information")
    col3, col4 = st.columns(2)
    with col3:
        monthly_income = st.number_input("Monthly Income *", min_value=0, value=50000, step=1000)
        designation = st.selectbox(
            "Designation *",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
            index=1,
        )
    with col4:
        occupation = st.selectbox(
            "Occupation *",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"],
        )
        own_car_label = st.selectbox("Own Car *", ["Yes", "No"])
        own_car = 1 if own_car_label == "Yes" else 0

    st.divider()
    st.subheader("🏖️ Travel Preferences")
    st.caption("Travel history and preferences")
    col5, col6 = st.columns(2)
    with col5:
        num_trips = st.number_input("Number of Trips *", min_value=0, max_value=50, value=2, step=1)
        passport_label = st.selectbox("Has Passport *", ["Yes", "No"])
        passport = 1 if passport_label == "Yes" else 0
    with col6:
        preferred_star = st.number_input("Preferred Property Star (1-5) *", min_value=1, max_value=5, value=4, step=1)
        product_pitched = st.selectbox(
            "Product Pitched *",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
            index=1,
        )

    st.divider()
    st.subheader("📊 Interaction Details")
    st.caption("Sales pitch and follow-up information")
    col7, col8 = st.columns(2)
    with col7:
        type_of_contact = st.selectbox("Type of Contact *", ["Self Enquiry", "Company Invited"])
        followups = st.number_input("Number of Follow-ups *", min_value=0, max_value=20, value=3, step=1)
    with col8:
        pitch_duration = st.number_input("Duration of Pitch (minutes) *", min_value=0, max_value=180, value=15, step=1)
        satisfaction = st.number_input("Pitch Satisfaction (1-5) *", min_value=1, max_value=5, value=4, step=1)

    submitted = st.form_submit_button("✨ Predict Purchase Probability", use_container_width=True)

if submitted:
    data = {
        'Age': float(age),
        'MonthlyIncome': float(monthly_income),
        'DurationOfPitch': float(pitch_duration),
        'NumberOfFollowups': float(followups),
        'PreferredPropertyStar': float(preferred_star),
        'NumberOfTrips': float(num_trips),
        'Passport': float(passport),
        'PitchSatisfactionScore': float(satisfaction),
        'OwnCar': float(own_car),
        'NumberOfChildrenVisiting': float(num_children),
        'TypeofContact': type_of_contact,
        'Occupation': occupation,
        'Gender': gender,
        'Designation': designation,
        'MaritalStatus': marital_status,
        'ProductPitched': product_pitched,
    }

    df = pd.DataFrame([data])

    # Exact preprocessing and prediction pipeline from predict.py
    X_num_scaled = scaler.transform(df[NUMERIC_FEATURES])
    X_cat_encoded = encoder.transform(df[CAT_FEATURES])
    X_processed = np.concatenate([X_cat_encoded, X_num_scaled], axis=1)
    X_pca = pca.transform(X_processed)

    prediction = int(model.predict(X_pca)[0])
    probabilities = model.predict_proba(X_pca)[0]
    buy_prob = float(probabilities[1]) * 100 if len(probabilities) > 1 else float(prediction) * 100

    result = "YES  - Likely to Buy" if prediction == 1 else "NO  - Not Likely to Buy"

    if prediction == 1:
        st.markdown(
            f'<div class="result-yes">✅ {result} (Probability: {buy_prob:.1f}%)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-no">⚠️ {result} (Probability: {buy_prob:.1f}%)</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("Powered by Machine Learning • Travel Industry Analytics")

