import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

# Load model and feature list
@st.cache_resource
def load_model():
    if not os.path.exists("best_house_price_model.pkl"):
        st.error("Model file not found. Please train and save the model as 'best_house_price_model.pkl'.")
        return None
    return joblib.load("best_house_price_model.pkl")

@st.cache_resource
def load_features():
    if not os.path.exists("model_features.pkl"):
        st.error("Feature list file not found. Please save the feature list as 'model_features.pkl'.")
        return None
    return joblib.load("model_features.pkl")

model = load_model()
feature_list = load_features()

# Load template row from train.csv
@st.cache_data
def get_template_row():
    df = pd.read_csv("hpp_data/train.csv")
    template = df.iloc[0].copy()
    if 'SalePrice' in template:
        template = template.drop('SalePrice')
    return template

template_row = get_template_row()

# User input fields
def user_input_features():
    st.sidebar.header("Input House Features")
    OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, int(template_row['OverallQual']))
    GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=int(template_row['GrLivArea']))
    TotalBsmtSF = st.sidebar.number_input("Total Basement SF", min_value=0, max_value=5000, value=int(template_row.get('TotalBsmtSF', 800)))
    FirstFlrSF = st.sidebar.number_input("1st Floor SF", min_value=0, max_value=5000, value=int(template_row.get('1stFlrSF', 800)))
    SecondFlrSF = st.sidebar.number_input("2nd Floor SF", min_value=0, max_value=5000, value=int(template_row.get('2ndFlrSF', 0)))
    YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=int(template_row.get('YearBuilt', 2000)))
    YearRemodAdd = st.sidebar.number_input("Year Remodeled", min_value=1800, max_value=2025, value=int(template_row.get('YearRemodAdd', 2000)))
    YrSold = st.sidebar.number_input("Year Sold", min_value=1800, max_value=2025, value=int(template_row.get('YrSold', 2010)))
    FullBath = st.sidebar.number_input("Full Bath", min_value=0, max_value=5, value=int(template_row.get('FullBath', 2)))
    HalfBath = st.sidebar.number_input("Half Bath", min_value=0, max_value=5, value=int(template_row.get('HalfBath', 0)))
    BsmtFullBath = st.sidebar.number_input("Basement Full Bath", min_value=0, max_value=5, value=int(template_row.get('BsmtFullBath', 0)))
    BsmtHalfBath = st.sidebar.number_input("Basement Half Bath", min_value=0, max_value=5, value=int(template_row.get('BsmtHalfBath', 0)))
    HasPool = st.sidebar.selectbox("Has Pool", [0, 1], format_func=lambda x: "Yes" if x else "No", index=int(template_row.get('PoolArea', 0) > 0))
    # Add more fields as needed
    data = {
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "TotalBsmtSF": TotalBsmtSF,
        "1stFlrSF": FirstFlrSF,
        "2ndFlrSF": SecondFlrSF,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "YrSold": YrSold,
        "FullBath": FullBath,
        "HalfBath": HalfBath,
        "BsmtFullBath": BsmtFullBath,
        "BsmtHalfBath": BsmtHalfBath,
        "HasPool": HasPool
    }
    return data

user_inputs = user_input_features()

# Prepare input for prediction
input_row = template_row.copy()
for k, v in user_inputs.items():
    if k in input_row:
        input_row[k] = v
# Engineered features
input_row['TotalSF'] = user_inputs['TotalBsmtSF'] + user_inputs['1stFlrSF'] + user_inputs['2ndFlrSF']
input_row['HouseAge'] = user_inputs['YrSold'] - user_inputs['YearBuilt']
input_row['RemodAge'] = user_inputs['YrSold'] - user_inputs['YearRemodAdd']
input_row['IsNew'] = int(user_inputs['YearBuilt'] == user_inputs['YrSold'])
input_row['TotalBath'] = user_inputs['FullBath'] + 0.5 * user_inputs['HalfBath'] + user_inputs['BsmtFullBath'] + 0.5 * user_inputs['BsmtHalfBath']
input_row['QualAreaInteraction'] = user_inputs['OverallQual'] * input_row['TotalSF']
input_row['PoolArea'] = 1 if user_inputs['HasPool'] == 1 else 0

# Polynomial features
poly_feats = ['TotalSF', 'GrLivArea', 'OverallQual', 'HouseAge']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = pd.DataFrame(poly.fit_transform(pd.DataFrame([{k: input_row[k] for k in poly_feats}])),
                        columns=poly.get_feature_names_out(poly_feats), index=[0])
for feat in poly_feats:
    if feat in input_row:
        del input_row[feat]
input_df = pd.concat([pd.DataFrame([input_row]), poly_data], axis=1)
input_df = input_df.loc[:, ~input_df.columns.duplicated()]

# Ensure all required columns are present and in the correct order
if feature_list is not None:
    for col in feature_list:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_list]

# Get dtypes from the original training data
train_df = pd.read_csv("hpp_data/train.csv")
if 'SalePrice' in train_df.columns:
    train_df = train_df.drop(columns=['SalePrice'])

for col in input_df.columns:
    if col in train_df.columns:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        else:
            input_df[col] = input_df[col].astype(str).fillna('None')
    else:
        # For engineered features, assume numeric
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

if model is not None and feature_list is not None:
    st.subheader("Input Summary")
    st.write(input_df)
    if st.button("Predict House Price"):
        prediction = model.predict(input_df)[0]
        try:
            pred_price = np.expm1(prediction)
        except Exception:
            pred_price = prediction
        pred_price_inr = pred_price * 83
        st.success(f"Estimated House Price: ₹{pred_price_inr:,.0f} (INR)")

    st.subheader("Model Feature Importance")
    try:
        importances = None
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
            st.bar_chart(imp_df.set_index("Feature"))
        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.info("Feature importance visualization not available.")