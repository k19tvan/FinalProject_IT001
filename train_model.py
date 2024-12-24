import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
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

one_hot_encodings = {}
for col in features_one_hot:
    one_hot_encodings[col] = sorted(X_cat[col].unique())

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

lr_pred = lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)

#Cross Validation

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    num_cols = list(X_num.columns)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print("Average Logistic Regression Accuracy (Cross Validation):", sum(accuracies) / len(accuracies))

# Shap

import shap
import numpy as np
import pandas as pd

feature_names = X.columns

explainer = shap.LinearExplainer(lr_model, X_train)
shap_values = explainer.shap_values(X_test)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
})

feature_importance = feature_importance.sort_values('importance', ascending=False)

feature_importance['importance_normalized'] = feature_importance['importance'] / feature_importance['importance'].max()

print("The impact of features on the likelihood of the target being 1:")
print(feature_importance)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)

pickle.dump(lr_model, open('lr_model.pkl', 'wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))
# pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))