import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# =========================
# 1. 加载模型、标准化器和特征列
# =========================
model_path = "svm_final_model.pkl"
scaler_path = "svm_scaler.pkl"
feature_cols_path = "svm_feature_columns.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(feature_cols_path)

# =========================
# 2. Streamlit 界面
# =========================
st.title("SVM predict dNCR")
st.write("Please enter patient characteristics：")

# 只有这4个特征需要标准化
scale_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']

input_data_original = {}

# 连续特征输入
for col in scale_cols:
    input_data_original[col] = st.number_input(f"{col}:", value=0.0)

# Education（连续但不标准化）
input_data_original['Education'] = st.selectbox(
    "Education (0=文盲, 1=小学, 2=初中, 3=高中, 4=大专及以上):", 
    options=list(range(0, 5))
)

# 二分类特征
for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
    input_data_original[feat] = st.selectbox(f"{feat}:", options=[0, 1])

# =========================
# 3. 预测按钮
# =========================
if st.button("predict"):
    # 构建输入数据
    input_data = {}
    
    # 连续特征（未标准化）
    for col in scale_cols:
        input_data[col] = input_data_original[col]
    
    # Education（连续但不标准化）
    input_data['Education'] = input_data_original['Education']
    
    # 二分类特征（drop_first=True，所以只有_1列）
    for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
        input_data[f'{feat}_1'] = input_data_original[feat]
    
    # 创建DataFrame
    X_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # 只标准化这4个特征
    X_input[scale_cols] = scaler.transform(X_input[scale_cols])
    
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Predicted probabilities: {pred_prob[0]:.4f}")
    st.write(f"Predicted results: {'yes' if pred_label[0] == 1 else 'no'}")

    # =========================
    # 创建背景数据
    # =========================
    background_data = {}
    
    # 连续特征设为0
    for col in scale_cols:
        background_data[col] = 0
    
    # Education 设为 0（文盲作为参考）
    background_data['Education'] = 0
    
    # 二分类特征设为 0（参考类别）
    for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
        background_data[f'{feat}_1'] = 0
    
    background_data = pd.DataFrame([background_data], columns=feature_cols)
    
    # 只标准化这4个特征
    background_data[scale_cols] = scaler.transform(background_data[scale_cols])

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # =========================
    # 显示 SHAP 值
    # =========================
    display_features = ['Age', 'Education', 'MOCA_Score', 'Operation_Time', 'GFR', 'Weakened', 'Depression', 'Nutritional_Risk']
    
    st.write("SHAP values of each feature：")
    for feat in display_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original[feat]})")
        elif f'{feat}_1' in feature_cols:
            idx = feature_cols.index(f'{feat}_1')
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original[feat]})")

    # =========================
    # 生成 force plot
    # =========================
    st.subheader("SHAP force plot of the prediction")
    
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
    
    st.components.v1.html(html_content, height=300, scrolling=False)
