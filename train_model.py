import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ['Age','MonthlyIncome','DurationOfPitch','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting']
CAT_FEATURES = ['TypeofContact','Occupation','Gender','Designation','MaritalStatus','ProductPitched']


df = pd.read_csv('Travel.csv')
df.dropna(inplace=True)
df['Gender'] = df['Gender'].replace({'Fe Male':'Female'})


X = df.drop(['ProdTaken', 'CustomerID'], axis=1)
y = df['ProdTaken']


scaler = StandardScaler()
scaler.fit(X[NUMERIC_FEATURES])
joblib.dump(scaler, 'scaler.joblib')

oe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
oe.fit(X[CAT_FEATURES])
joblib.dump(oe, 'encoder.joblib')

X_num_scaled = scaler.transform(X[NUMERIC_FEATURES])
X_cat_encoded = oe.transform(X[CAT_FEATURES])  # no .toarray()
X_processed = np.concatenate([X_cat_encoded, X_num_scaled], axis=1)


pca = PCA(n_components=0.9)
pca.fit(X_processed)
joblib.dump(pca, 'pca.joblib')


X_pca_final = pca.transform(X_processed) 
knn = KNeighborsClassifier(n_neighbors=3)
kn=knn.fit(X_pca_final, y)
joblib.dump(kn, 'knn_model.joblib')

print("All four deployment files saved successfully: scaler.joblib, encoder.joblib, pca.joblib, knn_model.joblib")