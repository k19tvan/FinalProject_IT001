import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Churn Prediction App", layout="wide")
page = st.sidebar.selectbox("Choose a page", ["Model Training", "Prediction"])

def load_and_preprocess_data(file):
    df = pd.read_csv(file)
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

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(one_hot_encodings, 'one_hot_encodings.pkl')

    return X, y, num_cols

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40, stratify=y)

    st.session_state.X_TRAIN = X_train
    model = LogisticRegression()
    
    with st.spinner('Training model...'):
        model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with st.spinner('Performing cross validation...'):
        cv_scores = cross_val_score(model, X, y, cv=5)
        avg_cv_score = np.mean(cv_scores)
    
    feature_importance = None
    shap_values = None
    X_display = None
    
    with st.spinner('Calculating SHAP values...'):
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Shapley Value': shap_values.mean(axis=0)
        })
        feature_importance = feature_importance.sort_values('Shapley Value', ascending=False)
        
        X_display = X_test

    joblib.dump(model, 'model.pkl')
    if feature_importance is not None:
        joblib.dump(feature_importance, 'feature_importance.pkl')
        joblib.dump(shap_values, 'shap_values.pkl')

    return test_acc, f1, avg_cv_score, feature_importance, shap_values, X_display

def user_input_features():
    st.subheader('Enter New Customer Information:')
    
    col1, col2 = st.columns(2)
    
    with col1:
        customerID = st.text_input('Customer ID')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', [0, 1], index=0)
        partner = st.selectbox('Partner', ['Yes', 'No'])
        dependents = st.selectbox('Dependents', ['Yes', 'No'])
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=36)
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['Yes', 'No'])
    
    with col2:
        online_backup = st.selectbox('Online Backup', ['Yes', 'No'])
        device_protection = st.selectbox('Device Protection', ['Yes', 'No'])
        tech_support = st.selectbox('Tech Support', ['Yes', 'No'])
        streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No'])
        streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', 
                                    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=1000.0)

    data = {
        'customerID': customerID,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return pd.DataFrame(data, index=[0])

def data_transform(df):

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].astype('float64')

    X_num = df.select_dtypes(include=['int64', 'float64'])
    X_cat = df.select_dtypes(include=['object'])
    X_cat = X_cat.drop("customerID", axis=1)

    try:
        label_encoders = joblib.load('label_encoders.pkl')
        one_hot_encodings = joblib.load('one_hot_encodings.pkl')

        features_one_hot = ["Contract", "PaymentMethod", "InternetService"]
        for col in X_cat.columns:
            if col not in features_one_hot:
                X_cat[col] = label_encoders[col].transform(X_cat[col])

        encoded_df = pd.DataFrame()
        for col in features_one_hot:
            for category in one_hot_encodings[col]:
                col_name = f"{col}_{category}"
                encoded_df[col_name] = (X_cat[col] == category).astype(int)
        
        X_cat = pd.concat([X_cat.drop(features_one_hot, axis=1), encoded_df], axis=1)
        X_cat = X_cat.astype(int)

        X = pd.concat([X_num, X_cat], axis=1)
        X = X.drop(["StreamingTV", "StreamingMovies", "MultipleLines", "PhoneService", "gender"], axis=1)

        scaler = joblib.load('scaler.pkl')
        num_cols = list(X_num.columns)
        X[num_cols] = scaler.transform(X[num_cols])
        
        return X

    except Exception as e:
        st.error(f"Error in transformation: {str(e)}")
        return None

def training_page():
    st.title('Customer Churn Model Training')
    
    model_type = 'Logistic Regression'  
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        with st.spinner('Loading and preprocessing data...'):
            X, y, num_cols = load_and_preprocess_data(uploaded_file)

        test_acc, f1, avg_cv_score, feature_importance, shap_values, X_display = train_model(X, y)
        
        st.success('Model trained successfully!')
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(label="Testing Accuracy", value=f"{test_acc:.4f}")
        
        with metrics_col2:
            st.metric(label="F1 Score", value=f"{f1:.4f}")
            
        with metrics_col3:
            st.metric(label="Avg Cross Validation Score", value=f"{avg_cv_score:.4f}")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Feature Importance')
            st.write(feature_importance[['Feature', 'Shapley Value']]) 

            st.markdown("""
            ### What is Shapley Value?
            - **Shapley Value**: Represents the degree of influence of each feature on the prediction. Positive values indicate that the feature promotes the likelihood of the event (churn), while negative values indicate that it reduces the likelihood of the event
            """)

        with col2:
            st.subheader('SHAP Summary Plot')
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_display, feature_names=X.columns, show=False)
            st.pyplot(fig)

            st.markdown("""
            ### How to read SHAP Summary Plot?
            - **SHAP Summay Plot**: Each point on the plot corresponds to a customer. The horizontal axis represents the SHAP value (influence) of the features, while the vertical axis lists the names of the features. The color indicates the value of the feature (low or high), making it easy to identify the influence of each feature on the prediction.
            """)

        st.session_state.model_trained = True

def prediction_page():
    st.title('Customer Churn Prediction')

    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.warning('Please train the model first on the Model Training page.')
        return

    user_data = user_input_features()

    if st.button('Predict Churn'):
        transformed_data = data_transform(user_data)
        if transformed_data is not None:
            try:
                model = joblib.load('model.pkl')
                prediction = model.predict(transformed_data)
                prediction_proba = model.predict_proba(transformed_data)
                
                st.subheader('Prediction Results')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction[0] == 1:
                        st.error('⚠️ Customer is likely to churn')
                    else:
                        st.success('✅ Customer is likely to stay')
                
                with col2:
                    st.metric(
                        label="Churn Probability",
                        value=f"{prediction_proba[0][1]:.2%}"
                    )
                
                with col3:
                    st.metric(
                        label="Stay Probability",
                        value=f"{prediction_proba[0][0]:.2%}"
                    )

                X_train = st.session_state.X_TRAIN 

                with st.spinner('Calculating SHAP values for the prediction...'):
                    feature_names = X_train.columns
                    explainer = shap.LinearExplainer(model, X_train)
                    shap_values_new = explainer.shap_values(transformed_data)
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Shapley Values': shap_values_new.mean(axis=0)
                    })

                    feature_importance = feature_importance.sort_values('Shapley Values', ascending=False)

                    st.write(feature_importance)

            except FileNotFoundError:
                st.error('Model files not found. Please train the model first.')

def main():
    if page == "Model Training":
        training_page()
    else:
        prediction_page()

if __name__ == '__main__':
    main()