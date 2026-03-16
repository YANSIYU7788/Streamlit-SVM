import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# =========================
# 1. 加载模型、标准化器和特征列（相对路径）
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

continuous_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']

# 定义原始分类特征
original_categorical = ['Education', 'Weakened', 'Depression', 'Nutritional_Risk']

# 用户输入原始值
input_data_original = {}
for col in continuous_cols:
    input_data_original[col] = st.number_input(f"{col}:", value=0.0)

for col in original_categorical:
    if col == 'Education':
        input_data_original[col] = st.selectbox(f"{col}:", options=list(range(0, 5)))
    else:
        input_data_original[col] = st.selectbox(f"{col}:", options=[0, 1])

# =========================
# 3. 预测按钮
# =========================
if st.button("predict"):
    # 初始化所有特征为0
    input_data = {col: 0 for col in feature_cols}
    
    # 设置连续特征
    for col in continuous_cols:
        if col in feature_cols:
            input_data[col] = input_data_original[col]
    
    # 设置分类特征（独热编码）
    for orig_col in original_categorical:
        user_value = input_data_original[orig_col]
        # 找到所有相关的独热编码列
        related_cols = [col for col in feature_cols if col.startswith(orig_col + '_')]
        for col in related_cols:
            # 提取类别值
            try:
                category = int(col.split('_')[-1])
                input_data[col] = 1 if user_value == category else 0
            except:
                pass
    
    # 创建DataFrame，确保列顺序与feature_cols一致
    X_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # 标准化连续特征
    X_input[continuous_cols] = scaler.transform(X_input[continuous_cols])
    
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Predicted probabilities: {pred_prob[0]:.4f}")
    st.write(f"Predicted results: {'yes' if pred_label[0] == 1 else 'no'}")

    # 创建背景数据
    background_data = pd.DataFrame([{col: 0 for col in feature_cols}], columns=feature_cols)

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # =========================
    # 聚合 SHAP 值到原始特征
    # =========================
    original_features = continuous_cols + original_categorical
    aggregated_shap = {}
    aggregated_values = {}

    for orig_feat in original_features:
        if orig_feat in continuous_cols:
            # 连续特征直接使用
            idx = feature_cols.index(orig_feat)
            aggregated_shap[orig_feat] = shap_values[idx]
            aggregated_values[orig_feat] = input_data_original[orig_feat]
        else:
            # 分类特征：聚合所有相关的独热编码列
            related_cols = [col for col in feature_cols if col.startswith(orig_feat + '_')]
            related_shap = sum([shap_values[feature_cols.index(col)] for col in related_cols])
            aggregated_shap[orig_feat] = related_shap
            aggregated_values[orig_feat] = input_data_original[orig_feat]

    # 显示原始特征的 SHAP 值
    st.write("SHAP values of each feature (original features)：")
    for name in original_features:
        st.write(f"{name}: {aggregated_shap[name]:.4f}")

    # 生成力图（使用原始特征）
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=np.array([aggregated_shap[f] for f in original_features]),
        features=np.array([aggregated_values[f] for f in original_features]),
        feature_names=original_features,
        matplotlib=False
    )

    shap.save_html("shap_force_plot.html", force_plot)

    st.subheader("SHAP force plot of the prediction")
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=400)
