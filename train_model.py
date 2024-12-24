import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('customer_churn.csv')

# Preprocessing

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].astype('float64')

X_v = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

X_num = X_v.select_dtypes(include=['int64', 'float64'])
X_cat = X_v.select_dtypes(include=['object'])
X_cat = X_cat.drop("customerID", axis=1)

label_encoders = {}
features_one_hot = ["Contract", "PaymentMethod", "InternetService"]

for col in X_cat.columns:
    if col not in features_one_hot:
        label_encoders[col] = LabelEncoder()
        X_cat[col] = label_encoders[col].fit_transform(X_cat[col])

X_cat = pd.get_dummies(X_cat, columns=features_one_hot, drop_first=False)
X_cat = X_cat.astype(int)

X = pd.concat([X_num, X_cat], axis=1)
X = X.drop(["StreamingTV", "StreamingMovies", "MultipleLines", "PhoneService", "gender"], axis=1)

scaler = StandardScaler()
num_cols = list(X_num.columns)
X[num_cols] = scaler.fit_transform(X[num_cols])


# Data train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40, stratify=y)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :", accuracy_lr)

#Cross Validation

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv_scores = cross_val_score(lr_model, X, y, cv=5)
avg_cv_score = np.mean(cv_scores)

print("Average Logistic Regression Accuracy (Cross Validation):", avg_cv_score)

# Shap

import shap
import numpy as np
import pandas as pd

feature_names = X.columns

explainer = shap.LinearExplainer(lr_model, X_train)
shap_values = explainer.shap_values(X_test)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'shap_value': (shap_values).mean(axis=0)
})

feature_importance = feature_importance.sort_values('shap_value', ascending=False)

print(feature_importance)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
