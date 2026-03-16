import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# =========================
# 1. Load model, scaler, and feature columns
# =========================
model_path = "svm_final_model.pkl"
scaler_path = "svm_scaler.pkl"
feature_cols_path = "svm_feature_columns.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(feature_cols_path)

# =========================
# 2. Streamlit UI
# =========================
st.title("SVM Prediction for dNCR")
st.write("Please enter patient characteristics:")

# Continuous features that need scaling
scale_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']

input_data_original = {}

# Continuous features input
for col in scale_cols:
    input_data_original[col] = st.number_input(f"{col}:", value=0.0)

# Education input (not scaled)
input_data_original['Education'] = st.selectbox(
    "Education:",
    options=list(range(0, 5)),
    format_func=lambda x: {
        0: "0 - Illiterate",
        1: "1 - Primary School",
        2: "2 - Middle School",
        3: "3 - High School",
        4: "4 - College or Above"
    }[x]
)

# Binary features
for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
    input_data_original[feat] = st.selectbox(
        f"{feat}:",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

# =========================
# 3. Prediction button
# =========================
if st.button("Predict"):
    # Prepare input data
    input_data = {}
    
    # Continuous features
    for col in scale_cols:
        input_data[col] = input_data_original[col]
    
    # Education
    input_data['Education'] = input_data_original['Education']
    
    # Binary features (drop_first=True in training)
    for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
        input_data[f'{feat}_1'] = input_data_original[feat]
    
    # Create DataFrame
    X_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # Scale the continuous features
    X_input[scale_cols] = scaler.transform(X_input[scale_cols])
    
    # Prediction
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Predicted probability: {pred_prob[0]:.4f}")
    st.write(f"Predicted result: {'Yes' if pred_label[0] == 1 else 'No'}")

    # =========================
    # Background data for SHAP
    # =========================
    background_data = {}
    
    # Continuous features set to 0
    for col in scale_cols:
        background_data[col] = 0
    
    # Education reference category
    background_data['Education'] = 0
    
    # Binary features reference category
    for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
        background_data[f'{feat}_1'] = 0
    
    background_data = pd.DataFrame([background_data], columns=feature_cols)
    
    # Scale continuous features
    background_data[scale_cols] = scaler.transform(background_data[scale_cols])

    # =========================
    # SHAP explainer
    # =========================
    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # =========================
    # Display SHAP values
    # =========================
    display_features = ['Age', 'Education', 'MOCA_Score', 'Operation_Time', 'GFR', 
                        'Weakened', 'Depression', 'Nutritional_Risk']
    
    st.write("SHAP values for each feature:")
    for feat in display_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original[feat]})")
        elif f'{feat}_1' in feature_cols:
            idx = feature_cols.index(f'{feat}_1')
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original[feat]})")

    # =========================
    # SHAP force plot
    # =========================
    st.subheader("SHAP Force Plot")
    
    shap_vals_list = []
    feature_vals_list = []
    feature_names_list = []
    
    for feat in display_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)
        elif f'{feat}_1' in feature_cols:
            idx = feature_cols.index(f'{feat}_1')
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)
    
        # =========================
    # SHAP force plot (Modified)
    # =========================
    st.subheader("SHAP Force Plot")

    # 确保 SHAP 值和特征名称的顺序一致
    shap_vals_list = []
    feature_vals_list = []
    feature_names_list = []

    for feat in display_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)
        elif f'{feat}_1' in feature_cols:
            idx = feature_cols.index(f'{feat}_1')
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)

       # =========================
   # SHAP force plot
    # =========================
    st.subheader("SHAP Force Plot")
    
    shap_vals_list = []
    feature_vals_list = []
    feature_names_list = []
    
    for feat in display_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)
        elif f'{feat}_1' in feature_cols:
            idx = feature_cols.index(f'{feat}_1')
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)
    
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=np.array(shap_vals_list),
        features=np.array(feature_vals_list),
        feature_names=feature_names_list,
        matplotlib=False
    )
    
    shap.save_html("shap_force_plot.html", force_plot)
    
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    st.components.v1.html(html_content, height=500, width=2000,  scrolling=False)
